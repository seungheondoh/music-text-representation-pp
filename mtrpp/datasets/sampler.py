import random
from torch.utils.data import Dataset
# https://www.jasonosajima.com/ns.html

class Sampler(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.freq = [len(d) for d in self.datasets]
        self.total = sum(self.freq)
        self.prob = [d.prob for d in self.datasets]
        print(self.freq, self.total, self.prob)

    def __getitem__(self, index):
        sampled_dataset = random.choices(population=self.datasets, weights=self.prob, k=1)[0]
        if index < len(sampled_dataset):
            return sampled_dataset[index]
        else:
            random_idx = random.randrange(0, len(sampled_dataset))
            return sampled_dataset[random_idx]

    def __len__(self):
        return sum(self.freq)