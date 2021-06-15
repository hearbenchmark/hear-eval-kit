import csv

import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    """
    Read in a CSV file, and return data rows as [column 1, (column 2, ...)]
    """

    def __init__(self, csv_file, labels_as_ints=False):
        # Our CSV files don't have headers, so we can't use
        # pd.read_csv
        # I'm on the fence whether we want CSV headers or not,
        # since the format is standardized.
        with open(csv_file) as f:
            csvreader = csv.reader(f)
            self.rows = [row for row in csvreader]
        ncol = len(self.rows[0])
        # Make sure all rows have the same number of column,
        # and rewrite as [col1, (col2, ...)]
        for idx, row in enumerate(self.rows):
            assert len(row) == ncol
            if labels_as_ints:
                self.rows[idx] = [row[0], torch.tensor([int(v) for v in row[1:]])]
            else:
                self.rows[idx] = [row[0], row[1:]]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]
