# merge_and_tokenize.py
import os
from transformers import BertTokenizerFast
import torch
from collections import Counter

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def read_bio_file(filepath):
    all_sentences, all_labels = [], []
    sentence, labels = [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    all_sentences.append(sentence)
                    all_labels.append(labels)
                    sentence, labels = [], []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    sentence.append(splits[0])
                    labels.append(splits[1])
    return all_sentences, all_labels

bio_files = [
    "c_cvpr1_BIO.txt", "c_cvpr2_BIO.txt", "c_cvpr3_BIO.txt", "c_cvpr4_BIO.txt", "c_cvpr5_BIO.txt", 
    "c_cvpr6_BIO.txt", "c_cvpr7_BIO.txt", "c_cvpr8_BIO.txt", "c_cvpr9_BIO.txt", "c_cvpr10_BIO.txt",
    "c_biophysics1_BIO.txt", "c_biophysics2_BIO.txt", "c_biophysics3_BIO.txt", "c_biophysics4_BIO.txt",
    "c_biophysics5_BIO.txt", "c_biophysics6_BIO.txt", "c_biophysics7_BIO.txt", "c_biophysics8_BIO.txt",
    "c_biophysics9_BIO.txt", "c_biophysics10_BIO.txt"
]

all_sents, all_tags = [], []
for file in bio_files:
    sents, tags = read_bio_file(file)
    all_sents.extend(sents)
    all_tags.extend(tags)

label2id = {"O": 0, "B-TERM": 1, "I-TERM": 2}
max_len = 128

encodings = tokenizer(
    all_sents,
    is_split_into_words=True,
    padding="max_length",
    truncation=True,
    max_length=max_len,
    return_tensors="pt"
)

labels = []
flat_label_ids = []
for i, label in enumerate(all_tags):
    word_ids = encodings.word_ids(batch_index=i)
    label_ids = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            mapped = label2id.get(label[word_idx], 0)
            label_ids.append(mapped)
            flat_label_ids.append(mapped)
        else:
            mapped = label2id.get(label[word_idx], 0)
            label_ids.append(mapped)
            flat_label_ids.append(mapped)
        previous_word_idx = word_idx
    labels.append(label_ids)


labels = torch.tensor(labels)

dataset = {
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"],
    "labels": labels
}

torch.save(dataset, "c_merged_cvpr_biophysics.pt")
print("\n✅ Saved to c_merged_cvpr_biophysics.pt")
