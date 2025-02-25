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
            df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            log_in_local_rank_0(">>> 'date' column not found in CSV. A synthetic date column has been added.")

        # Log that we are using the entire CSV as test data.
        log_in_local_rank_0(
            f'>>> Using entire dataset as test data (indices 0 to {len(df)-1}).'
        )

        # Select only the numeric columns (excluding 'date').
        numeric_cols = [col for col in df.columns if col != 'date']
        df_values = df[numeric_cols].values

        # Scale the entire dataset.
        scaler = StandardScaler()
        scaler.fit(df_values)
        scaled_data = scaler.transform(df_values)

        # Transpose so that each sequence becomes a row.
        self.hf_dataset = scaled_data.transpose(1, 0)
        self.num_sequences = len(self.hf_dataset)
        self.window_length = self.context_length + self.prediction_length

        # Build a list of (sequence_index, offset) tuples for valid windows.
        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            # Allow windows that end at index n_points (hence n_points+1)
            for offset_idx in range(self.window_length, n_points + 1):
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
            'inputs': window_seq[: self.context_length],
            'labels': window_seq[self.context_length:],
        }

