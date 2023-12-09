from torch.utils.data import Dataset


def load_data_file(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            row = []
            count = 0
            for number in line.split():
                if 2 <= count <= 4:
                    row.append(float(number) * 1e6) # from m to um
                elif 5 <= count <= 7:
                    row.append(float(number) * 1e15) # from N to fN
                count += 1
            data.append(row)
        return np.array(data)


class GODataset(Dataset):
    def __init__(self, r, f):
        self.r = r
        self.f = f

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        return (self.r[i].astype(np.float32), self.f[i].astype(np.float32))
