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

    x = data[:,0:-1] # input data

    number_samples = data.shape[0]
    y = reshape(data[:, -1], (number_samples, 1)) # targets

    return (x, y)