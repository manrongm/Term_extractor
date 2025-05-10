import os
import sys
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast

# 模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from bert_bilstm_crf import BERT_BiLSTM_CRF

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = "bert-base-cased"
num_tags = 3
id2tag = {0: "O", 1: "B-TERM", 2: "I-TERM"}

# 加载模型和 tokenizer
model = BERT_BiLSTM_CRF(bert_model_name=bert_model_name, num_tags=num_tags).to(device)
model.load_state_dict(torch.load("model_cvpr_biophysics1.pt", map_location=device))
model.eval()
tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

# 输入文本
test_sentences_path = "processed_cvpr/cvpr1_sentences.txt"
with open(test_sentences_path, encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

def decode_terms(tokens, tags):
    """
    还原术语：合并 BERT 分词中的 subword（##开头）
    """
    terms = []
    current = []
    for token, tag in zip(tokens, tags):
        if tag == "B-TERM":
            if current:
                terms.append(" ".join(current))
            current = [token]
        elif tag == "I-TERM" and current:
            current.append(token)
        else:
            if current:
                terms.append(" ".join(current))
                current = []
    if current:
        terms.append(" ".join(current))

    # 去除##号，并合并subword
    clean_terms = []
    for term in terms:
        tokens = term.split()
        detok = []
        for tok in tokens:
            if tok.startswith("##") and detok:
                detok[-1] += tok[2:]
            else:
                detok.append(tok)
        clean_terms.append(" ".join(detok))
    return clean_terms

# 推理
for idx, sentence in enumerate(sentences[:30]):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    tags = [id2tag[p] for p in predictions]
    terms = decode_terms(tokens[1:-1], tags[1:-1])  # 去除[CLS], [SEP]

    print(f"🔹 Sample #{idx+1}")
    print(f"🔍 Extracted Terms: {terms}\n")
