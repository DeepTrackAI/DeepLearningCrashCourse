def load_data_1d(filename):
    import csv
    from numpy import asarray
    
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
        data = asarray(data).astype(float)

    x = data[:, 0] # input
    y = data[:, 1] # output / targets / groundtruth
    
    return (x, y)

def load_data(filename):
    import csv
    from numpy import asarray, reshape
    
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
        data = asarray(data).astype(float)

    x = data[:, 0:-1] # input

    num_samples = data.shape[0]
    y = reshape(data[:, -1], (num_samples, 1)) # output / targets / groundtruth
    
    return (x, y)