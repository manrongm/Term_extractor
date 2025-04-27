import torch
from transformers import BertTokenizerFast
from model.bert_bilstm_crf import BERT_BiLSTM_CRF

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = "bert-base-cased"  # or your fine-tuned version
num_tags = 3
model = BERT_BiLSTM_CRF(bert_model_name, num_tags)
model.load_state_dict(torch.load("model_cvpr_biophysics1.pt", map_location=device))
model = model.to(device)
model.eval()

# load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

# load data
dataset = torch.load("test_bert_dataset.pt")  # test set
input_ids = dataset["input_ids"].to(device)
attention_mask = dataset["attention_mask"].to(device)

# prediction
with torch.no_grad():
    predictions = model(input_ids, attention_mask)

terms = []

# iterate each sentence
for i in range(len(predictions)):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
    tags = predictions[i]
    current_term = ""
    collected_terms = []

    for token, tag in zip(tokens, tags):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if tag == 1:  # B-TERM
            if current_term:
                collected_terms.append(current_term.strip())
            current_term = token
        elif tag == 2:  # I-TERM
            if token.startswith("##"):
                current_term += token[2:]
            else:
                current_term += " " + token
        else:  # O
            if current_term:
                collected_terms.append(current_term.strip())
                current_term = ""

    if current_term:
        collected_terms.append(current_term.strip())

    terms.extend(collected_terms)

# clean
terms = list(dict.fromkeys(terms))

# output
print("\n✅ extracted terms are:")
for term in terms:
    print("-", term)
