"""Module with functions to load data."""
import csv
from numpy import asarray


def load_data_1d(filename):
    """Load 1D data."""
    with open(filename) as file:
        reader = csv.reader(file)
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
        data = asarray(data).astype(float)
    x = data[:, 0]  # Input
    y = data[:, 1]  # Output/target/ground truth
    return (x, y)
