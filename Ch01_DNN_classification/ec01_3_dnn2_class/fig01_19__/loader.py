"""Module with functions to load data."""
import csv
from numpy import asarray, reshape


def load_data_1d(filename):
    """Load 1D data."""
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
        data = asarray(data).astype(float)  
    x = data[:, 0]  # Input.
    y = data[:, 1]  # Output/target/ground truth.
    return (x, y)


def load_data(filename):
    """Load multidimensional data."""
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
        data = asarray(data).astype(float)  
    x = data[:, 0:-1]  # Input.
    num_samples = data.shape[0]
    y = reshape(data[:, -1], (num_samples, 1))  # Output/target/ground truth.
    return (x, y)