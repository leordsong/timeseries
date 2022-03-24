from typing import List, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class WindowGenerator():

    def __init__(self,
                 input_width: int, label_width: int, offset: int,
                 train_df, val_df, test_df,
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
        
        self.train_time = train_df.pop(timestamp_column)
        self.val_time = val_df.pop(timestamp_column)
        self.test_time = test_df.pop(timestamp_column)
        
        self.train_mean = train_df.mean()
        self.train_std = train_df.std()

        train_df = (train_df - self.train_mean) / self.train_std
        val_df = (val_df - self.train_mean) / self.train_std
        test_df = (test_df - self.train_mean) / self.train_std

        # Store the raw data
        self.train_df = train_df
        self.train_index = np.arange(len(train_df)).reshape(-1, 1)
        self.val_df = val_df
        self.val_index = np.arange(len(val_df)).reshape(-1, 1)
        self.test_df = test_df
        self.test_index = np.arange(len(test_df)).reshape(-1, 1)

        # Work out the label column indices.
        self.input_columns = input_columns
        self.label_columns = label_columns
        self.timestamp_column = timestamp_column
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.columns = train_df.columns
        self.column_indices = {name: i for i, name in enumerate(self.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.offset = offset

        self.total_window_size = input_width + offset

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        indices = features[:, :, -1]
        features = features[:, :, :-1]
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.input_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.input_columns],
                axis=-1
            )
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, len(self.input_columns if self.input_columns else self.columns)])
        labels.set_shape([None, self.label_width, len(self.label_columns if self.label_columns else self.columns)])
        indices.set_shape([None, self.total_window_size])

        return inputs, indices, labels
    
    def make_dataset(self, data, indices, batch, shuffle=True):
        data = np.array(data, dtype=np.float32)
        data = np.hstack([data, indices])

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch
        )

        ds = ds.map(self.split_window)

        return ds

    def train(self, batch=32):
        return self.make_dataset(self.train_df, self.train_index, batch)

    def val(self, batch=32):
        return self.make_dataset(self.val_df, self.val_index, batch)

    def test(self, batch=32):
        return self.make_dataset(self.test_df, self.test_index, batch, False)

    def plot(self, model, plot_col='packetLossRate', max_subplots=3):
        inputs, indices, labels = next(iter(self.test().take(1)))
        indices = indices.numpy().astype('int32')
        
        plt.figure(figsize=(16, 8))
        plot_col_index = self.column_indices[plot_col]
        input_col_index = self.input_columns_indices.get(plot_col, None) if self.input_columns else plot_col_index
        label_col_index = self.label_columns_indices.get(plot_col, None) if self.label_columns else plot_col_index
        assert input_col_index is not None and label_col_index is not None
        max_n = min(max_subplots, len(inputs))

        name = None
        interval = 5
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(plot_col)
            sub_indices = indices[n, ::interval]
            if n == 0:
                name = self.test_time[sub_indices][0].strftime("%Y-%m-%d %H:%M:%S")
            sub_times = [x.strftime("%M:%S.%f")[:-4] for x in self.test_time[sub_indices]]
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
                plt.scatter(self.label_indices, values,
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time ' + name + ' [s]')

    def plot_y(self, model, plot_col='packetLossRate', max_subplots=3):
        
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        input_col_index = self.input_columns_indices.get(plot_col, None) if self.input_columns else plot_col_index
        label_col_index = self.label_columns_indices.get(plot_col, None) if self.label_columns else plot_col_index
        assert input_col_index is not None and label_col_index is not None

        batch_size = None
        name = None
        n = 0
        interval = 5
        for inputs, indices, labels in self.test().take(max_subplots):
            batch_size = inputs.shape[0]
            plt.subplot(max_subplots, 1, n+1)
            plt.ylabel(plot_col)
            sub_indices = tf.squeeze(indices[:, -self.label_width]).numpy().astype('int32')[::interval]
            if n == 0:
                name = self.test_time[sub_indices][sub_indices[0]].strftime("%Y-%m-%d %H:%M:%S")
            sub_times = [x.strftime("%M:%S.%f")[:-4] for x in self.test_time[sub_indices]]
            sub_ticks = np.arange(batch_size)[::interval]
            plt.xticks(sub_ticks, sub_times)

            values = labels[:, -self.label_width, label_col_index] * self.train_std[plot_col] + self.train_mean[plot_col]
            plt.plot(np.arange(batch_size), values,
                        label='Inputs', marker='.', zorder=-10)
            if model is not None:
                predictions = model(inputs)
                values = predictions[:, -self.label_width, label_col_index] * self.train_std[plot_col] + self.train_mean[plot_col]
                plt.scatter(np.arange(batch_size), values,
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()
            n += 1

        plt.xlabel('Time ' + name + ' [s]')

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Input column name(s): {self.input_columns if self.input_columns else self.columns}',
            f'Label column name(s): {self.label_columns if self.label_columns else self.columns}'
        ])
