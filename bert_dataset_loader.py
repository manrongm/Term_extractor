import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class BertTermDataset(Dataset):
    def __init__(self, tensor_data, indices=None):
        if indices is None:
            self.input_ids = tensor_data["input_ids"]
            self.attention_mask = tensor_data["attention_mask"]
            self.labels = tensor_data["labels"]
        else:
            self.input_ids = tensor_data["input_ids"][indices]
            self.attention_mask = tensor_data["attention_mask"][indices]
            self.labels = tensor_data["labels"][indices]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# 🔁 Load full tensor data
dataset = torch.load("c_merged_cvpr_biophysics.pt")

# Split indices manually
total_size = dataset["input_ids"].shape[0]
train_size = int(0.8 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(42)
all_indices = torch.randperm(total_size, generator=generator).tolist()

train_indices = all_indices[:train_size]
val_indices = all_indices[train_size:train_size+val_size]
test_indices = all_indices[train_size+val_size:]

# ✅ Create base datasets
train_dataset = BertTermDataset(dataset, indices=train_indices)
val_dataset = BertTermDataset(dataset, indices=val_indices)
test_dataset = BertTermDataset(dataset, indices=test_indices)

# # 🔁 查找增强样本：包含 B/I 标签的样本
# aug_indices = [idx for idx in train_indices
#                if 1 in dataset["labels"][idx] or 2 in dataset["labels"][idx]]
# aug_dataset = BertTermDataset(dataset, indices=aug_indices)

# # ✅ 拼接增强集
# train_dataset = ConcatDataset([train_dataset, aug_dataset])
# print(f"🔁 Augmented {len(aug_dataset)} samples added to training set")

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

