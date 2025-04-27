# bert_dataset_loader.py

import torch
from torch.utils.data import Dataset, DataLoader, random_split

class BertTermDataset(Dataset):
    def __init__(self, tensor_data):
        self.input_ids = tensor_data["input_ids"]
        self.attention_mask = tensor_data["attention_mask"]
        self.labels = tensor_data["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# load data
dataset = torch.load("merged_cvpr_biophysics.pt")
full_dataset = BertTermDataset(dataset)

# split 80% train, 15% val, 5% test
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # fix the random seed
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
