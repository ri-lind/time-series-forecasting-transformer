#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port),
                            rank=rank, world_size=world_size)


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


# ------------------ Base Metric ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val
        self.count = 0  # count of elements (tokens)

    def push(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs):
        n = preds.numel()
        self.count += n
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


# ------------------ RMSE Metric ------------------
class RMSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        # Sum squared error for this push
        return torch.sum((preds - labels) ** 2)

    def compute(self):
        # Compute RMSE from accumulated squared error and count
        mse = self.value / self.count
        return torch.sqrt(mse)


# ------------------ MAE Metric ------------------
class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))

    def compute(self):
        return self.value / self.count


# ------------------ MAPE Metric ------------------
class MAPEMetric(SumEvalMetric):
    def __init__(self, name, init_val: float = 0.0, epsilon: float = 1e-8):
        super().__init__(name, init_val)
        self.epsilon = epsilon

    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs((preds - labels) / (labels + self.epsilon)))

    def compute(self):
        # This returns the mean absolute percentage error (as a fraction)
        return self.value / self.count


# ------------------ R2 Metric ------------------
class R2Metric:
    def __init__(self, name):
        self.name = name
        self.sse = 0.0        # Sum of squared errors: sum((labels - preds)^2)
        self.sum_y = 0.0      # Sum of labels
        self.sum_y2 = 0.0     # Sum of labels squared
        self.count = 0        # Number of elements

    def push(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs):
        self.sse += torch.sum((labels - preds) ** 2)
        self.sum_y += torch.sum(labels)
        self.sum_y2 += torch.sum(labels ** 2)
        self.count += labels.numel()

    def compute(self):
        # Compute total sum of squares (SST)
        sst = self.sum_y2 - (self.sum_y ** 2) / self.count
        # Avoid division by zero
        if sst == 0:
            return torch.tensor(0.0)
        return 1 - (self.sse / sst)


# ------------------ Explained Variance Metric ------------------
class ExplainedVarianceMetric:
    def __init__(self, name):
        self.name = name
        self.sum_errors = 0.0   # Sum of (labels - preds)
        self.sum_error2 = 0.0   # Sum of squared errors
        self.sum_y = 0.0        # Sum of labels
        self.sum_y2 = 0.0       # Sum of labels squared
        self.count = 0          # Number of elements

    def push(self, preds: torch.Tensor, labels: torch.Tensor, **kwargs):
        errors = labels - preds
        self.sum_errors += torch.sum(errors)
        self.sum_error2 += torch.sum(errors ** 2)
        self.sum_y += torch.sum(labels)
        self.sum_y2 += torch.sum(labels ** 2)
        self.count += labels.numel()

    def compute(self):
        # Variance of errors:
        var_errors = (self.sum_error2 - (self.sum_errors ** 2) / self.count) / self.count
        # Variance of labels:
        var_y = (self.sum_y2 - (self.sum_y ** 2) / self.count) / self.count
        if var_y == 0:
            return torch.tensor(0.0)
        return 1 - (var_errors / var_y)


class TimeMoE:
    def __init__(self, model_path, device, context_length, prediction_length, **kwargs):
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
                trust_remote_code=True,
            )

        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length

        outputs = model.generate(
            inputs=batch['inputs'].to(device).to(model.dtype),
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]
        labels = batch['labels'].to(device)
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels


import numpy as np
import matplotlib.pyplot as plt

def plot_performance(plot_name: str, input, preds, labels):
    plt.figure(figsize=(10, 5))
    
    # Convert to numpy if tensors
    if isinstance(input, torch.Tensor):
        input = input.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Define the x-axis for past values and forecast
    x_input = np.arange(len(input))
    x_forecast = np.arange(len(input), len(input) + len(labels))
    
    # Plot past values using their natural indices
    plt.plot(x_input, input, label="Past Values", marker="o")
    
    # Plot ground truth and forecast starting after past values
    plt.plot(x_forecast, labels, label="Ground Truth", marker="o")
    plt.plot(x_forecast, preds, label="Forecast", marker="x")
    
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Forecast vs. Ground Truth")
    plt.legend()
    
    plot_path = f"{plot_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    
    return plot_path



def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_length = args.prediction_length
    plot_name = args.plot_name
    
    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', 9899)
    world_size = int(os.getenv('WORLD_SIZE') or 1)
    rank = int(os.getenv('RANK') or 0)
    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    if torch.cuda.is_available():
        try:
            setup_nccl(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
            device = f"cuda:{local_rank}"
            is_dist = True
        except Exception as e:
            print('Error: ', f'Setup nccl fail, so set device to cpu: {e}')
            device = 'cpu'
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    # ------------------ Metrics ------------------
    mse_metric = RMSEMetric(name='rmse')  # Will compute RMSE
    mae_metric = MAEMetric(name='mae')
    mape_metric = MAPEMetric(name='mape')
    r2_metric = R2Metric(name='r2')
    expl_var_metric = ExplainedVarianceMetric(name='explained_variance')

    metric_list = [mse_metric, mae_metric, mape_metric, r2_metric, expl_var_metric]

    acc_count = 0  # For RMSE and MAE we count tokens

    model = TimeMoE(
        args.model,
        device,
        context_length=context_length,
        prediction_length=prediction_length
    )
    dataset = BenchmarkEvalDataset(
        args.data,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    if torch.cuda.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
    else:
        sampler = None
    test_dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        drop_last=False,
    )
    plotted = False # for 
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            preds, labels = model.predict(batch)
            # (Assume preds and labels have matching shapes.)
            input = batch["inputs"].to(model.device).to(model.model.dtype)
            # insert plotting method
            if not plotted:
                plot_performance(plot_name, input, preds, labels)
                plotted = True
            mse_metric.push(preds, labels)
            mae_metric.push(preds, labels)
            mape_metric.push(preds, labels)
            r2_metric.push(preds, labels)
            expl_var_metric.push(preds, labels)
            acc_count += count_num_tensor_elements(preds)

    # Gather distributed statistics.
    # For RMSE, MAE, and MAPE we need: mse_metric.value, mae_metric.value, mape_metric.value, and token count.
    # For R², we need: r2_metric.sse, r2_metric.sum_y, r2_metric.sum_y2, and r2_metric.count.
    # For Explained Variance, we need: expl_var_metric.sum_error2, expl_var_metric.sum_errors,
    # expl_var_metric.sum_y, expl_var_metric.sum_y2, and expl_var_metric.count.
    metric_tensors = [
        mse_metric.value, 
        mae_metric.value, 
        mape_metric.value, 
        torch.tensor(acc_count, device=model.device),
        r2_metric.sse, 
        r2_metric.sum_y, 
        r2_metric.sum_y2, 
        torch.tensor(r2_metric.count, device=model.device),
        expl_var_metric.sum_error2, 
        expl_var_metric.sum_errors, 
        expl_var_metric.sum_y, 
        expl_var_metric.sum_y2, 
        torch.tensor(expl_var_metric.count, device=model.device)
    ]
    if is_dist:
        stat_tensor = torch.stack(metric_tensors)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = torch.stack(metric_tensors)

    if rank == 0:
        item = {
            'model': args.model,
            'data': args.data,
            'context_length': args.context_length,
            'prediction_length': args.prediction_length,
        }
        # For RMSE, MAE, and MAPE: denominator is total token count.
        count = all_stat[3].item()
        rmse = torch.sqrt(all_stat[0] / count).item()
        mae = (all_stat[1] / count).item()
        mape = (all_stat[2] / count).item()

        # For R²:
        sse = all_stat[4].item()
        sum_y = all_stat[5].item()
        sum_y2 = all_stat[6].item()
        count_r2 = all_stat[7].item()
        sst = sum_y2 - (sum_y ** 2) / count_r2 if count_r2 > 0 else 0
        r2 = 1 - (sse / sst) if sst != 0 else 0

        # For Explained Variance:
        sum_error2 = all_stat[8].item()
        sum_errors = all_stat[9].item()
        sum_y_ex = all_stat[10].item()
        sum_y2_ex = all_stat[11].item()
        count_ex = all_stat[12].item()
        var_errors = (sum_error2 - (sum_errors ** 2) / count_ex) / count_ex if count_ex > 0 else 0
        var_y = (sum_y2_ex - (sum_y_ex ** 2) / count_ex) / count_ex if count_ex > 0 else 0
        explained_variance = 1 - (var_errors / var_y) if var_y != 0 else 0

        item[mse_metric.name] = rmse
        item[mae_metric.name] = mae
        item[mape_metric.name] = mape
        item[r2_metric.name] = r2
        item[expl_var_metric.name] = explained_variance
        
        print(f"Context Length: {context_length}")
        print(f"Prediction Length: {prediction_length}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"MAPE: {mape}")
        print(f"R²: {r2}")
        print(f"Explained Variance: {explained_variance}")

        logging.info(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Maple728/TimeMoE-50M',
        help='Model path'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Benchmark data path'
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=32,
        help='Batch size of evaluation'
    )
    parser.add_argument(
        '--context_length', '-c',
        type=int,
        help='Context length'
    )
    parser.add_argument(
        '--prediction_length', '-p',
        type=int,
        default=96,
        help='Prediction length'
    )
    
    parser.add_argument(
        '--plot_name', '-n',
        type=str,
        default="",
        help='Name of Plot to be created'
    )
    args = parser.parse_args()
    if args.context_length is None:
        if args.prediction_length == 96:
            args.context_length = 512
        elif args.prediction_length == 192:
            args.context_length = 1024
        elif args.prediction_length == 336:
            args.context_length = 2048
        elif args.prediction_length == 720:
            args.context_length = 3072
        else:
            args.context_length = args.prediction_length * 4
    evaluate(args)
