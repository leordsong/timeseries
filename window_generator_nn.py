from typing import List, Optional

import numpy as np
import torch as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta


class TimeseriesDataset(Dataset):

    def __init__(self, data: pd.DataFrame, window_size:int, input_slice:slice, label_slice:slice,
                 timestamp_column:pd.Series, input_columns:Optional[List[str]]=None, label_columns:Optional[List[str]]=None,
                 transform=None, target_transform=None):
        assert len(data) >= window_size
        self.data = data

        self.window_size = window_size
        self.input_slice = input_slice
        self.label_slice = label_slice
        
        self.time_column = timestamp_column
        self.indices = np.array(self.data.index)

        self.input_columns = input_columns
        self.label_columns = label_columns
        self.columns = data.columns
        self.column_indices = {name: i for i, name in enumerate(self.columns)}

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        indices = self.indices[idx: idx + self.window_size]
        values = self.data.loc[indices].values.astype('float32')
        inputs = values[self.input_slice]
        labels = values[self.label_slice]
        if self.input_columns is not None:
            inputs = np.stack(
                [inputs[:, self.column_indices[name]] for name in self.input_columns],
                axis=-1
            )
        if self.label_columns is not None:
            labels = np.stack(
                [labels[:, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        
        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            labels = self.target_transform(labels)
        return inputs, labels, indices



class WindowGenerator():

    def __init__(self,
                 input_width: int, label_width: int, offset: int,
                 train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                 input_columns:Optional[List[str]]=None, label_columns:Optional[List[str]]=None,
                 timestamp_column:str='timestamp'):
        """Create a new instance of WindowGenerator class

        Args:
            input_width (int): Length of input sequence.
            label_width (int): Length of output sequence.
            offset (int): Offset between input and output.
            train_df (DataFrame): The data for training.
            val_df (DataFrame): The data for validation.
            test_df (DataFrame): The data for testing.
            input_columns (Optional[List[str]], optional): Input column names. Defaults to None.
            label_columns (Optional[List[str]], optional): Output column names. Defaults to None.
            timestamp_column (str, optional): Name of the timestamp column. Defaults to 'timestamp'.
        """

        train_time = train_df.pop(timestamp_column)
        val_time = val_df.pop(timestamp_column)
        test_time = test_df.pop(timestamp_column)
        
        self.train_mean = train_df.mean()
        self.train_std = train_df.std()

        train_df = (train_df - self.train_mean) / self.train_std
        val_df = (val_df - self.train_mean) / self.train_std
        test_df = (test_df - self.train_mean) / self.train_std
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.offset = offset

        self.total_window_size = input_width + offset + label_width
        assert self.total_window_size > input_width
        assert self.total_window_size > label_width
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # Columns
        self.columns = train_df.columns
        self.column_indices = {name: i for i, name in enumerate(self.columns)}
        self.input_columns = input_columns
        self.label_columns = label_columns
        self.timestamp_column = timestamp_column
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        # Store the dataset
        self.train_data = TimeseriesDataset(train_df, self.total_window_size, self.input_indices,
                                            self.labels_slice, train_time, input_columns, label_columns)
        self.valid_data = TimeseriesDataset(val_df, self.total_window_size, self.input_indices,
                                            self.labels_slice, val_time, input_columns, label_columns)
        self.test_data = TimeseriesDataset(test_df, self.total_window_size, self.input_indices,
                                            self.labels_slice, test_time, input_columns, label_columns)

    def train(self, batch=32, shuffle=True):
        return DataLoader(self.train_data, batch_size=batch, shuffle=shuffle)

    def val(self, batch=32, shuffle=True):
        return DataLoader(self.valid_data, batch_size=batch, shuffle=shuffle)

    def test(self, batch=32, shuffle=False):
        return DataLoader(self.test_data, batch_size=batch, shuffle=shuffle)

    def plot(self, model, plot_col='packetLossRate', max_subplots=3):
        inputs, labels, indices = next(iter(self.test(batch=max_subplots)))
        indices = indices.numpy().astype('int32')
        
        plt.figure(figsize=(16, 8))
        plot_col_index = self.column_indices[plot_col]
        input_col_index = self.input_columns_indices.get(plot_col, None) if self.input_columns else plot_col_index
        label_col_index = self.label_columns_indices.get(plot_col, None) if self.label_columns else plot_col_index
        assert input_col_index is not None and label_col_index is not None

        name = None
        interval = 5
        for n in range(max_subplots):
            plt.subplot(max_subplots, 1, n+1)
            plt.ylabel(plot_col)
            sub_indices = indices[n, ::interval]
            if n == 0:
                name = self.test_data.time_column.loc[sub_indices][sub_indices[0]].strftime("%Y-%m-%d %H:%M:%S")
            sub_times = [x.strftime("%M:%S.%f")[:-4] for x in self.test_data.time_column[sub_indices]]
            sub_ticks = np.arange(self.total_window_size)[::interval]
            plt.xticks(sub_ticks, sub_times)
            values = inputs[n, :, input_col_index] * self.train_std[plot_col] + self.train_mean[plot_col]
            plt.plot(self.input_indices, values,
                    label='Inputs', marker='.', zorder=-10)

            values = labels[n, :, label_col_index] * self.train_std[plot_col] + self.train_mean[plot_col]
            plt.plot(self.label_indices, values,
                     label='Labels', c='#2ca02c')
            if model is not None:
                predictions = model(inputs)
                values = predictions[n, :, label_col_index] * self.train_std[plot_col] + self.train_mean[plot_col]
                values = values.detach().numpy()
                plt.scatter(self.label_indices, values,
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time ' + name + ' [s]')

    def plot_y(self, model, plot_col='packetLossRate', max_subplots=3, width=50):
        
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        input_col_index = self.input_columns_indices.get(plot_col, None) if self.input_columns else plot_col_index
        label_col_index = self.label_columns_indices.get(plot_col, None) if self.label_columns else plot_col_index
        assert input_col_index is not None and label_col_index is not None

        batch_size = None
        name = None
        interval = 5
        dataloader_iterator = iter(self.test(batch=width))
        for i in range(max_subplots):
            inputs, labels, indices = next(dataloader_iterator)
            batch_size = inputs.shape[0]
            plt.subplot(max_subplots, 1, i+1)
            plt.ylabel(plot_col)
            sub_indices = nn.squeeze(indices[:, -self.label_width]).numpy().astype('int32')[::interval]
            if i == 0:
                name = self.test_data.time_column[sub_indices][sub_indices[0]].strftime("%Y-%m-%d %H:%M:%S")
            sub_times = [x.strftime("%M:%S.%f")[:-4] for x in self.test_data.time_column[sub_indices]]
            sub_ticks = np.arange(batch_size)[::interval]
            plt.xticks(sub_ticks, sub_times)

            values = labels[:, -self.label_width, label_col_index] * self.train_std[plot_col] + self.train_mean[plot_col]
            plt.plot(np.arange(batch_size), values,
                        label='Inputs', marker='.', zorder=-10)
            if model is not None:
                predictions = model(inputs)
                values = predictions[:, -self.label_width, label_col_index] * self.train_std[plot_col] + self.train_mean[plot_col]
                values = values.detach().numpy()
                plt.scatter(np.arange(batch_size), values,
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if i == 0:
                plt.legend()

        plt.xlabel('Time ' + name + ' [s]')

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Input column name(s): {self.input_columns if self.input_columns else self.columns}',
            f'Label column name(s): {self.label_columns if self.label_columns else self.columns}'
        ])
