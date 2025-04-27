import os
import sys
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast
from sklearn.metrics import precision_score, recall_score, f1_score

# model path
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from bert_bilstm_crf import BERT_BiLSTM_CRF

# data loading
from bert_dataset_loader import val_loader

# --------------------------
# tool functions
# --------------------------
def evaluate_terms(predicted_terms, gold_terms):
    gold_set = set(gold_terms)
    predicted_set = set(predicted_terms)
    y_true = [1 if term in gold_set else 0 for term in predicted_terms]
    y_pred = [1] * len(predicted_terms)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

def bio_to_terms(words, labels):
    terms = []
    current_term = []
    for word, label in zip(words, labels):
        if label == "B-TERM":
            if current_term:
                terms.append(" ".join(current_term))
                current_term = []
            current_term.append(word)
        elif label == "I-TERM":
            if current_term:
                current_term.append(word)
        else:  # "O"
            if current_term:
                terms.append(" ".join(current_term))
                current_term = []
    if current_term:
        terms.append(" ".join(current_term))
    return terms

# --------------------------
# setting
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = "bert-base-cased"
num_tags = 3

tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

id2tag = {
    0: "O",
    1: "B-TERM",
    2: "I-TERM"
}

# --------------------------
# loading model
# --------------------------
model = BERT_BiLSTM_CRF(bert_model_name=bert_model_name, num_tags=num_tags).to(device)
model.load_state_dict(torch.load("model_cvpr_biophysics1.pt", map_location=device))
model.eval()

# --------------------------
# evaluation
# --------------------------
all_gold_terms = []
all_predicted_terms = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Running Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size, seq_len = input_ids.shape

        predictions = model(input_ids, attention_mask)

        for i in range(batch_size):
            pred_tags = predictions[i]
            gold_tags = labels[i].cpu().tolist()

            valid_len = attention_mask[i].sum().item()
            pred_tags = pred_tags[:valid_len]
            gold_tags = gold_tags[:valid_len]
            input_ids_valid = input_ids[i][:valid_len].cpu().tolist()

            words = tokenizer.convert_ids_to_tokens(input_ids_valid)

            clean_words, clean_pred_tags, clean_gold_tags = [], [], []
            for word, p_tag, g_tag in zip(words, pred_tags, gold_tags):
                if word not in tokenizer.all_special_tokens:
                    clean_words.append(word)
                    clean_pred_tags.append(p_tag)
                    clean_gold_tags.append(g_tag)

            pred_labels = [id2tag[idx] for idx in clean_pred_tags]
            gold_labels = [id2tag[idx] if idx in id2tag else "O" for idx in clean_gold_tags]

            # Recover terms
            pred_terms = bio_to_terms(clean_words, pred_labels)
            gold_terms = bio_to_terms(clean_words, gold_labels)

            all_predicted_terms.extend(pred_terms)
            all_gold_terms.extend(gold_terms)

# --------------------------
# result
# --------------------------
precision, recall, f1 = evaluate_terms(all_predicted_terms, all_gold_terms)

print("\n📈 Evaluation Results (based on real tokens):")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# --------------------------
# save terms
# --------------------------
with open("predicted_terms.txt", "w", encoding="utf-8") as f:
    for term in all_predicted_terms:
        f.write(term + "\n")
print("✅ Saved predicted terms to 'predicted_terms.txt'.")
