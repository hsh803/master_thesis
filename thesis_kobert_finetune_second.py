import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Import KoBERT model and Tokenizer
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

kobert_model = BertModel.from_pretrained('skt/kobert-base-v1')
print("KoBERT Model loaded successfully.")
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
print("KoBERT Tokenizer loaded successfully.")

# Data setup for labeled data
def data_setup_label(dataset):
  data_dict = {'text': [], 'label': []}
  with open(dataset, 'r', encoding='utf-8') as file:
    data = file.read().strip().split('\n')
    for i in data:
      i = i.split('\t')
      data_dict['text'].append(i[0])
      data_dict['label'].append(int(i[1]))

  return data_dict


# Define dataset class for Detection task

class DetectionDataset(Dataset):
  def __init__(self, texts, labels, tokenizer):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = self.texts[idx]
    encoding = self.tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].squeeze(0).to(device),
      'attention_mask': encoding['attention_mask'].squeeze(0).to(device),
      'label': torch.tensor(self.labels[idx], dtype=torch.long).to(device)
      }


# Define a multi-task model

class MultiTaskModel(nn.Module):
  def __init__(self, kobert_model, num_label):
    super(MultiTaskModel, self).__init__()
    self.encoder = kobert_model
    self.detection_head = nn.Linear(self.encoder.config.hidden_size, num_label)
    self.dropout = nn.Dropout(0.1)

  def forward(self, input_ids, attention_mask):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state
    pooled_output = self.dropout(hidden_states[:, 0, :])

    return self.detection_head(pooled_output)



train_files = [
    "synthetic_data_kobert_5170.txt",
    "synthetic_data_kobert_7190.txt",
    "synthetic_data_kobert_91100.txt",
    "synthetic_data_kobert_91100_big.txt",
    "synthetic_data_kobert_combined.txt"
]

test = data_setup_label("test_data.txt")
test_dataset = DetectionDataset(test['text'], test['label'], tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


num_label = 2
num_epoch = 1

# Training loop

for train_file in train_files:
  print(f"\n=== Training on {train_file} ===")
  suffix = os.path.splitext(os.path.basename(train_file))[0]

  train = data_setup_label(train_file)
  train_dataset = DetectionDataset(train['text'], train['label'], tokenizer)
  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

  mymodel = MultiTaskModel(kobert_model, num_label)
  model_path = "/proj/uppmax2024-2-24/hanna/kobert_finetune_v1.pth"
  mymodel.load_state_dict(torch.load(model_path, map_location=device), strict=False)
  mymodel.to(device)


  optimizer = torch.optim.AdamW([
      {'params': mymodel.encoder.parameters(), 'lr': 3e-5},
      {'params': mymodel.detection_head.parameters(), 'lr': 5e-5}
  ], weight_decay=0.01)
  criterion = nn.CrossEntropyLoss(reduction='mean')

  mymodel.train()
  for epoch in range(num_epoch):

   # Training on labeled data for Detection
    total_loss = 0
    correct_prediction = 0
    total_prediction = 0

    for label_batch in train_loader:

      # Labeled Data for detection
      label_input_ids = label_batch['input_ids'].to(device)
      label_attention_mask = label_batch['attention_mask'].to(device)
      label = label_batch['label'].to(device)

      # Forward Pass for both tasks and mlm
      detection_logits = mymodel(label_input_ids, label_attention_mask)

      # Compute loss for Detection
      detection_loss = criterion(detection_logits, label)

      # Backward Pass and Optimization
      optimizer.zero_grad()
      detection_loss.backward()
      optimizer.step()

      total_loss += detection_loss.item()

      predictions = torch.argmax(detection_logits, dim=1)
      correct_prediction += (predictions == label).sum().item()
      total_prediction += label.size(0)

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = (correct_prediction / total_prediction) * 100
    print(f"[MTL] Epoch {epoch+1}/{num_epoch}, Avg Loss: {avg_loss:.2f}, Avg Accuracy: {avg_accuracy}")

  # Save the fine-tuned KoBERT model
  model_save_path = f'/proj/uppmax2024-2-24/hanna/{suffix}_finetune.pth'
  torch.save(mymodel.state_dict(), model_save_path)

  print(f"Model saved at: {model_save_path}")

  # Test model

  mymodel = MultiTaskModel(kobert_model, num_label)
  mymodel.load_state_dict(torch.load(model_save_path, map_location=device))
  mymodel.to(device)
  mymodel.eval()

  result = []

  with torch.no_grad():
    correct = 0
    total = 0

    for batch in test_loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      label = batch['label'].to(device)
      text = batch['text']

      outputs = mymodel(input_ids, attention_mask)
      predictions = torch.argmax(outputs, dim=1)
      correct += (predictions == label).sum().item()
      total += label.size(0)
      for i in range(len(text)):
        result.append([text[i], predictions[i], label[i]])

    accuracy = (correct / total) * 100
    print(f"[Test] Accuracy: {accuracy:.3f}")

  # Save result to a text file
  test_save_path = f"/proj/uppmax2024-2-24/hanna/{suffix}_test_result.txt"

  # Save result to a text file
  with open(test_save_path, 'w', encoding='utf-8') as f_result:
    for data in result:
      text, prediction, label = data
      f_result.write(f"{text}\t{prediction}\t{label}\n")

  # Calculate precision, recall, F1

  all_predictions = [data[1].cpu().item() for data in result]
  all_labels = [data[2].cpu().item() for data in result]

  precision = precision_score(all_labels, all_predictions)
  recall = recall_score(all_labels, all_predictions)
  f1 = f1_score(all_labels, all_predictions)

  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")
  print(f"F1 Score: {f1:.3f}")

  print("Precision, Recall, F1 Scores are done.")

  print("Prediction distribution:", Counter(all_predictions))
  print("Label distribution:", Counter(all_labels))

  # Compute metrics
  print("Classification Report:")
  print(classification_report(all_labels, all_predictions, digits=3))

