import os
import sys
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast

# model path
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from bert_bilstm_crf import BERT_BiLSTM_CRF
from bert_dataset_loader import test_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = "bert-base-cased"
num_tags = 3  # O, B-TERM, I-TERM
id2tag = {0: "O", 1: "B-TERM", 2: "I-TERM"}

# load model & tokenizer
model = BERT_BiLSTM_CRF(bert_model_name=bert_model_name, num_tags=num_tags).to(device)
model.load_state_dict(torch.load("c_0.7_1e-5_model_cvpr_biophysics2.pt", map_location=device))
model.eval()

tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

def bio_to_terms(tokens, tags):
    terms = []
    current = []
    current_tag = None

    # merge subword token
    merged_tokens = []
    merged_tags = []

    for i in range(len(tokens)):
        token = tokens[i]
        tag = tags[i]

        if token.startswith("##") and merged_tokens:
            merged_tokens[-1] += token[2:]
        else:
            merged_tokens.append(token)
            merged_tags.append(tag)

    for token, tag in zip(merged_tokens, merged_tags):
        if tag == "B-TERM":
            if current:
                terms.append(" ".join(current))
            current = [token]
            current_tag = "B-TERM"
        elif tag == "I-TERM" and current_tag in {"B-TERM", "I-TERM"}:
            current.append(token)
            current_tag = "I-TERM"
        else:
            if current:
                terms.append(" ".join(current))
                current = []
                current_tag = None

    if current:
        terms.append(" ".join(current))
    return set(terms)

def iou_score(p, g):
    p_set, g_set = set(p.split()), set(g.split())
    intersection = p_set & g_set
    union = p_set | g_set
    return len(intersection) / len(union) if union else 0

def get_iou_hits(pred_terms, gold_terms, threshold=0.5):
    correct = 0
    matched_gold = set()
    for p in pred_terms:
        for g in gold_terms:
            if g in matched_gold:
                continue
            if iou_score(p, g) >= threshold:
                correct += 1
                matched_gold.add(g)
                break
    return correct


correct = 0
total_pred = 0
total_gold = 0
preview_limit = 20
previewed = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        predictions = model(input_ids, attention_mask)

        for i in range(len(predictions)):
            pred_ids = predictions[i]
            gold_ids = labels[i].cpu().tolist()
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids[i])

            pred_tags = [id2tag.get(id_, "O") for id_ in pred_ids]
            gold_tags = [id2tag.get(id_, "O") for id_ in gold_ids if id_ != -100]
            tokens = input_tokens[1:len(gold_tags)+1]
            pred_tags = pred_tags[1:len(gold_tags)+1]

            pred_terms = bio_to_terms(tokens, pred_tags)
            gold_terms = bio_to_terms(tokens, gold_tags)

            # correct += len(pred_terms & gold_terms)
            # correct += get_partial_hits(pred_terms, gold_terms)
            correct += get_iou_hits(pred_terms, gold_terms, threshold=0.5)
            total_pred += len(pred_terms)
            total_gold += len(gold_terms)

            if previewed < preview_limit:
                print(f"\n🔹 Sample #{previewed + 1}")
                print("🟦 Input Sentence:", " ".join(tokens))
                print("✅ Gold Terms    :", list(gold_terms))
                print("🔍 Pred Terms    :", list(pred_terms))
                previewed += 1

# metrics
precision = correct / total_pred if total_pred > 0 else 0
recall = correct / total_gold if total_gold > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n✅ Test Precision: {precision:.4f}")
print(f"✅ Test Recall: {recall:.4f}")
print(f"✅ Test F1-score: {f1:.4f}")
