class WindowGenerator():

    def __init__(self,
                 input_width, label_width, shift=0,
                 train_df=None, val_df=None, test_df=None,
                 input_columns=None, label_columns=None, timestamp_column='timestamp'):

        # Store the raw data
        self.train_df = train_df
        self.train_index = np.arange(len(train_df)).reshape(-1, 1)
        self.train_time = train_df.pop(timestamp_column)
        self.val_df = val_df
        self.val_index = np.arange(len(val_df)).reshape(-1, 1)
        self.val_time = val_df.pop(timestamp_column)
        self.test_df = test_df
        self.test_index = np.arange(len(test_df)).reshape(-1, 1)
        self.test_time = test_df.pop(timestamp_column)

        # Work out the label column indices.
        self.input_columns = input_columns
        self.label_columns = label_columns
        self.timestamp_column = timestamp_column
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift + label_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
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
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, len(self.input_columns)])
        labels.set_shape([None, self.label_width, len(self.label_columns)])

        return inputs, labels
    
    def make_dataset(self, data, shuffle=True, indices=None):
        data = np.array(data, dtype=np.float32)
        ## TODO indices

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=1
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df, False)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Input column name(s): {self.input_columns}',
            f'Label column name(s): {self.label_columns}'
        ])
