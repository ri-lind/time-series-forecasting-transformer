#!/usr/bin/env python
# -*- coding:utf-8 _*-
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from time_moe.utils.log_util import log_in_local_rank_0


class BenchmarkEvalDataset(Dataset):

    def __init__(self, csv_path, context_length: int, prediction_length: int):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        # Read the CSV file.
        df = pd.read_csv(csv_path)

        # If the CSV doesn't contain a 'date' column, add one.
        if 'date' not in df.columns:
            # Generate a date range. Adjust the start date and frequency as needed.
            df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            log_in_local_rank_0(">>> 'date' column not found in CSV. A synthetic date column has been added.")

        # Instead of splitting the dataset into train/validation/test with 70/20/10 ratios,
        # use the entire dataset as test data.
        border1s = [0, 0, 0]
        border2s = [len(df), len(df), len(df)]

        # Log splitting information using the date column.
        if 'date' in df.columns:
            start_dt = df.iloc[border1s[2]]['date']
            eval_start_dt = df.iloc[border1s[2] + context_length]['date']
            end_dt = df.iloc[border2s[2] - 1]['date']
            log_in_local_rank_0(
                f'>>> Split test data from {start_dt} to {end_dt}, '
                f'and evaluation start date is: {eval_start_dt}'
            )
        else:
            log_in_local_rank_0(
                f'>>> No date column found. Test data split indices: {border1s[2]} to {border2s[2]-1}, '
                f'evaluation starts at index {border1s[2] + context_length}'
            )

        # Select only the numeric columns (excluding 'date').
        numeric_cols = [col for col in df.columns if col != 'date']
        df_values = df[numeric_cols].values

        # In this modified version, the entire dataset is used as test data.
        test_data = df_values[border1s[2]:border2s[2]]

        # Scaling: fit on the whole dataset and transform.
        scaler = StandardScaler()
        scaler.fit(test_data)
        scaled_test_data = scaler.transform(test_data)

        # Assignment: transpose so that each sequence becomes a row.
        self.hf_dataset = scaled_test_data.transpose(1, 0)
        self.num_sequences = len(self.hf_dataset)
        self.window_length = self.context_length + self.prediction_length

        # Build a list of (sequence_index, offset) tuples for valid windows.
        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]
        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)
        return {
            'inputs': np.array(window_seq[: self.context_length], dtype=np.float32),
            'labels': np.array(window_seq[-self.prediction_length:], dtype=np.float32),
        }
