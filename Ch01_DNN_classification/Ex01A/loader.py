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

    x = data[:, 0] # input data
    y = data[:, 1] # output data / targets / ground truth
    
    return (x, y)