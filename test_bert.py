import os
import sys
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# 添加 model 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from bert_bilstm_crf import BERT_BiLSTM_CRF

# 加载 test loader
from bert_dataset_loader import test_loader

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型配置
bert_model_name = "bert-base-cased"
num_tags = 3  # O, B-TERM, I-TERM
model = BERT_BiLSTM_CRF(bert_model_name=bert_model_name, num_tags=num_tags).to(device)

# 加载最好的模型参数
model.load_state_dict(torch.load("model_cvpr_biophysics1.pt", map_location=device))
model.eval()

# Mapping from tag id to tag name
id2tag = {0: "O", 1: "B-TERM", 2: "I-TERM"}

# --------------------------------------------------------
# 辅助函数：将 BIO 标签转成术语短语列表
def bio_to_terms(tokens, tags):
    terms = []
    term = []
    for token, tag in zip(tokens, tags):
        if tag == "B-TERM":
            if term:
                terms.append(" ".join(term))
                term = []
            term.append(token)
        elif tag == "I-TERM" and term:
            term.append(token)
        else:
            if term:
                terms.append(" ".join(term))
                term = []
    if term:
        terms.append(" ".join(term))
    return terms

# --------------------------------------------------------

all_gold_terms = []
all_pred_terms = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        predictions = model(input_ids, attention_mask)  # list of [seq_len] predictions

        for i in range(len(predictions)):
            # original tokens
            tokens = input_ids[i]
            tokens = tokens.cpu().numpy().tolist()

            # label id to tags
            gold_ids = labels[i].cpu().numpy().tolist()
            pred_ids = predictions[i]

            gold_tags = [id2tag[id_] for id_ in gold_ids if id_ != -100]
            pred_tags = [id2tag[id_] for id_ in pred_ids]

            gold_terms = bio_to_terms(["token"] * len(gold_tags), gold_tags)
            pred_terms = bio_to_terms(["token"] * len(pred_tags), pred_tags)

            all_gold_terms.append(set(gold_terms))
            all_pred_terms.append(set(pred_terms))

# Precision / Recall / F1

correct = 0
total_pred = 0
total_gold = 0

for gold, pred in zip(all_gold_terms, all_pred_terms):
    correct += len(gold & pred)
    total_pred += len(pred)
    total_gold += len(gold)

precision = correct / total_pred if total_pred > 0 else 0
recall = correct / total_gold if total_gold > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

print(f"✅ Test Precision: {precision:.4f}")
print(f"✅ Test Recall: {recall:.4f}")
print(f"✅ Test F1-score: {f1:.4f}")
