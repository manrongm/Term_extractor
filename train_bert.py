import os
import sys
import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
from bert_bilstm_crf import BERT_BiLSTM_CRF
from bert_dataset_loader import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model_name = "bert-base-cased"
num_tags = 3
model = BERT_BiLSTM_CRF(bert_model_name=bert_model_name, num_tags=num_tags).to(device)

checkpoint_path = "c_备用_model_cvpr_biophysics2.pt"
# checkpoint_path = "---"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Warm-start: Loaded existing model weights")
else:
    print("Training from scratch")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

total_steps = len(train_loader) * 10
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

num_epochs = 10
best_val_loss = float("inf")
early_stop_patience = 3
no_improve_epochs = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == -100] = 0

        optimizer.zero_grad()
        loss = model(input_ids, attention_mask, labels_for_loss)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

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

    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     no_improve_epochs = 0
    torch.save(model.state_dict(), "c_热启动不增强0.7_1e-5_model_cvpr_biophysics2.pt")
        # print("✅ New best model saved.")
    # else:
    #     no_improve_epochs += 1
    #     print(f"No improvement. {no_improve_epochs}/{early_stop_patience} patience.")

    if no_improve_epochs >= early_stop_patience:
        print("Early stopping triggered.")
        break

print("Training completed.")
