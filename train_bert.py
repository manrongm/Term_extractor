import os
import sys
import torch
from tqdm import tqdm

# add model path
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from bert_bilstm_crf import BERT_BiLSTM_CRF

# data loading path
from bert_dataset_loader import train_loader, val_loader

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model setting up
bert_model_name = "bert-base-cased"
num_tags = 3  # O, B-TERM, I-TERM
model = BERT_BiLSTM_CRF(bert_model_name=bert_model_name, num_tags=num_tags).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# training arguments
num_epochs = 20
best_val_loss = float("inf")

# training epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == -100] = 0

        loss = model(input_ids, attention_mask, labels_for_loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
    
    # validations
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            labels_for_loss = labels.clone()
            labels_for_loss[labels_for_loss == -100] = 0

            loss = model(input_ids, attention_mask, labels_for_loss)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

    # save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "model_cvpr_biophysics2.pt")
        print("✅ New best model saved.")

print("🎉 Training completed.")
