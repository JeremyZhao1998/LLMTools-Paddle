import os
import json

from paddle.io import Dataset


class FinetuneDataset(Dataset):

    def __init__(self, data_root, dataset_name, split='train', subset_size=None):
        super().__init__()
        self.file_path = str(os.path.join(data_root, dataset_name))
        self.split = split
        self.subset_size = subset_size
        self.file_names = [file_name for file_name in os.listdir(self.file_path)
                           if file_name.endswith('.json') and split in file_name]
        self.data = []
        for file_name in self.file_names:
            if self.subset_size is not None and len(self.data) >= self.subset_size:
                break
            with open(os.path.join(self.file_path, file_name), 'r') as f:
                for line in f:
                    self.data.append(json.loads(line)["text"])
                    if self.subset_size is not None and len(self.data) >= self.subset_size:
                        break

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
