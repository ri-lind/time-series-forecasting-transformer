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


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        raise NotImplementedError


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


class MASEMetric(SumEvalMetric):
    def __init__(self, name, init_val: float = 0.0):
        super().__init__(name, init_val)
        self.num_sequences = 0

    def push(self, preds, labels, **kwargs):
        # For MASE, we count the number of sequences (i.e. batch size)
        batch_size = labels.shape[0]
        self.num_sequences += batch_size
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        # Assume labels has shape (batch, prediction_length)
        batch_size, pred_len = labels.shape
        abs_errors = torch.abs(preds - labels)         # shape: (batch, prediction_length)
        mae_per_seq = torch.mean(abs_errors, dim=1)      # (batch,)
        # Compute naive forecast error per sequence as the mean absolute difference
        # between consecutive values in the ground-truth labels.
        if pred_len < 2:
            naive_error = torch.ones(batch_size, device=labels.device)
        else:
            naive_error = torch.mean(torch.abs(labels[:, 1:] - labels[:, :-1]), dim=1)
        mase = mae_per_seq / (naive_error + 1e-8)        # (batch,)
        return torch.sum(mase)


class TimeMoE:
    def __init__(self, model_path, device, context_length, prediction_length, **kwargs):
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
                context_length=context_length,
                prediction_length=prediction_length
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
        self.context_length = context_length
        
        self.model.context_length = context_length
        self.model.prediction_length = prediction_length
        
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


def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_length = args.prediction_length

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

    # evaluation metrics: note that for MSE and MAE we count tokens, but for MASE we count sequences.
    mse_metric = MSEMetric(name='mse')
    mae_metric = MAEMetric(name='mae')
    mase_metric = MASEMetric(name='mase')
    metric_list = [mse_metric, mae_metric, mase_metric]

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

    acc_count = 0  # for MSE and MAE (total number of tokens)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            preds, labels = model.predict(batch)

            mse_metric.push(preds, labels)
            mae_metric.push(preds, labels)
            mase_metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

    # For MSE and MAE, average error per token; for MASE, average error per sequence.
    ret_metric = {}
    ret_metric[mse_metric.name] = mse_metric.value / acc_count
    ret_metric[mae_metric.name] = mae_metric.value / acc_count
    ret_metric[mase_metric.name] = mase_metric.value / mase_metric.num_sequences

    print(f'{rank} - {ret_metric}')

    metric_tensors = [mse_metric.value, mae_metric.value, mase_metric.value, acc_count]
    if is_dist:
        stat_tensor = torch.tensor(metric_tensors).to(model.device)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = metric_tensors

    if rank == 0:
        item = {
            'model': args.model,
            'data': args.data,
            'context_length': args.context_length,
            'prediction_length': args.prediction_length,
        }
        count = all_stat[-1]
        # Use appropriate counts for each metric
        item[mse_metric.name] = float(all_stat[0] / count)
        item[mae_metric.name] = float(all_stat[1] / count)
        item[mase_metric.name] = float(all_stat[2] / mase_metric.num_sequences)
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
