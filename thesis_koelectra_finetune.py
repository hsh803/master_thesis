import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           # Assign a variable for running the training model in GPU when it is available
print(device)

# Initialize KoELECTRA generator

from transformers import ElectraModel, ElectraTokenizer

koelectra_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")   # KoELECTRA-Base-v3
print("KoELECTRA Model loaded successfully.")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")   # Tokenizer add [CLS] and [SEP] tokens automatically by default when encoding
print("KoELECTRA Tokenizer loaded successfully.")

# Data setup for labeled data
def data_setup_label(dataset):
  data_dict = {'text': [], 'label': []}
  with open(dataset, 'r', encoding='utf-8') as file:
    data = file.read().strip()   # Remove trailing newlines
    data = data.split('\n')
    for i in data:
      i = i.split('\t')   # each text anc its label get splitted and saved as a string.
      data_dict['text'].append(i[0])
      data_dict['label'].append(int(i[1]))   # Change each label data point from string to integer (when using split, each line was saved as string, that is wny labels are stored as string)
                                             # Convert str to int is crucial due to converting to tensor when encoding the labels
  return data_dict

train = data_setup_label("train_data.txt")
test = data_setup_label("test_data.txt")

# Data setup for unlabeled data
def data_setup_unlabel(dataset):
  data_list = []
  with open(dataset, 'r') as file:
    data = file.read().strip()
    data = data.split('\n')
    for i in data:
      data_list.append(i)
  return data_list

unlabel = data_setup_unlabel("unlabel_data.txt")

# Define dataset class for Detection task

from torch.utils.data import Dataset

class DetectionDataset(Dataset):
  def __init__(self, texts, labels, tokenizer):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.texts)   # DataLoader needs it to determine the number of batches

  def __getitem__(self, idx): # idx helps fetch corresponding text and label from self.texts and self.labels. Each call to __getitem__ returns a single data point(input tokens, attention masks, label)
    text = self.texts[idx]   # Store original text
    encoding = self.tokenizer(
        text,
        truncation=True,
        max_length=256,       # BERT-based model has fixed max-length of input tokens, 512. Thus, impossible use dynamic padding function (collante_fn with pad_sequence)
        padding='max_length',
        return_tensors='pt'   # 'pt' stands for Pytorch. Returns tokenized outputs as tensors
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].squeeze(0).to(device),
      'attention_mask': encoding['attention_mask'].squeeze(0).to(device),
      'label': torch.tensor(self.labels[idx], dtype=torch.long).to(device)      # Convert label (integer: 0 or 1) to tensor because the model require tensor input. Neural network in Pytorch operates on tensors.
                                                                                # torch.long (torch.int64) is a data type representing a 64-bit integers. It is used to store integer values, especially class labels. If we use torch.float, we will get error.
      }

# Initialize train, valid, test dataset
train_dataset = DetectionDataset(train['text'], train['label'], tokenizer)
test_dataset = DetectionDataset(test['text'], test['label'], tokenizer)

# Define dataset class for Auto-labeling task
class AutoLabelingDataset(Dataset):
  def __init__(self, texts, tokenizer):
    self.texts = texts
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
        'attention_mask': encoding['attention_mask'].squeeze(0).to(device)
    }

unlabel_dataset = AutoLabelingDataset(unlabel, tokenizer)

# DataLoader

from torch.utils.data import DataLoader

# Define batch size
batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # each batch is a list of dictionaries containing input text and label as well as tensors of input_ids, attention_mask of each data point from Detecion- and AutoLanlingDataset class
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

unlabel_loader = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=True)

# Define a multi-task model

class MultiTaskModel(nn.Module):
  def __init__(self, koelectra_model, num_label):
    super(MultiTaskModel, self).__init__()
    self.encoder = koelectra_model   # a pre-trained model, KoELECTRA
    self.detection_head = nn.Linear(self.encoder.config.hidden_size, num_label)   # 768: A Transformer model has a hidden size, which represents the number of neurons in each hidden state vector of the tranformer. We can access this using: self.encoder.config.hidden_size
    self.mlm_head = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.vocab_size)
    self.dropout = nn.Dropout(0.1)

  def forward(self, input_ids, attention_mask, task):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)    # The model learns to output a probability distribution over classes. Index 0 corresponds to class 0 (e.g., "non-hate speech") and index 1 corresponds to class 1 (e.g., "hate speech")
                                                                                  # The model typically outputs logits which are raw, unnormalized socres and can be any real number, positive or negative, and they don’t sum to 1

    hidden_states = outputs.last_hidden_state     # outputs.last_hidden_state: the final hidden state output from a transformer encoder. Shape: (batch_size, seq_length, hidden_dim)
                                                           # [:, 0, :]: Selects all batch samples, Selects the first token which is [CLS],  Selects all hidden features for that token
    pooled_output = self.dropout(hidden_states[:, 0, :])            # Apply dropout to the pooled output to prevent overfitting during training

    if task == 'detection':
      return self.detection_head(pooled_output)    # Outputs: logits with a tensor of shape, (batch_size, num_label) in batch. For example, there are batch number of [0, 0] in [[0, 0], [0, 0], ...[0, 0]])

    elif task == 'mlm':
      return self.mlm_head(hidden_states)   # Ouputs: logits of a tensor of shape, (batch_size, seq_len, vocab_size)

num_label = 2
mymodel = MultiTaskModel(koelectra_model, num_label)
mymodel.to(device)

print("Ready to train")

# MLM Mask function

def masked_tokens(input_ids, tokenizer, mlm_probability=0.15):   # mlm_probability : 15% of tokens will be masked
  device = input_ids.device
  input_ids = input_ids.clone()
  labels = input_ids.clone()   # original input tokens are defined as labels

  probability_matrix = torch.full(labels.shape, mlm_probability, device=device)   # Create matrix of the same shape as the original input with value of mlm_probability
  special_tokens_masks = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]   # Identify special tokens like [CLS], [SEP], [PAD] using tokenizer
  probability_matrix.masked_fill_(torch.tensor(special_tokens_masks, dtype=torch.bool, device=device), value=0.0)   # Set the mask probability for special tokens to 0

  masked_indices = torch.bernoulli(probability_matrix).bool()   # Use Bernoulli sampling to decide which token will be masked (True if masked, otherwise False)
  labels[~masked_indices] = -100   # Set non-masked token to -100 so CrossEntropyLoss ignore them

  indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices   # 80% of the time, replace masked tokens with [MASK]
  input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)   # Replace masked tokens with [MASK] tokens ID

  indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced   # 10% of the time (half of remaining 20%), we replace masked tokens with random token IDs
  random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)   # Generate radom token IDs from voca
  input_ids[indices_random] = random_words[indices_random]   # Replace selected tokens with random random ones

  return input_ids, labels   # label Shape: (batch_size, seq_len)

# Training loop
import os

optimizer = torch.optim.AdamW([
    {'params': mymodel.encoder.parameters(), 'lr': 3e-5},
    {'params': mymodel.detection_head.parameters(), 'lr': 5e-5},
    {'params': mymodel.mlm_head.parameters(), 'lr': 1e-5},
], weight_decay=0.01)
criterion = nn.CrossEntropyLoss(reduction='mean')
num_epoch = 2
mlm_weight = 0.001  # Starting weight for auto-labeling loss

# Create an empty list to store pseudo labels and confidence score
pseudo_labeled_data = []

for epoch in range(num_epoch):
  mymodel.train()   # Set model to training mode

  # Training on labeled data for Detection
  total_loss = 0
  correct_prediction = 0
  total_prediction = 0

  for label_batch, unlabel_batch in zip(train_loader, unlabel_loader):
    
    # Labeled Data
    label_input_ids = label_batch['input_ids'].to(device)
    label_attention_mask = label_batch['attention_mask'].to(device)
    label = label_batch['label'].to(device)

    # Unlabeled Data
    unlabel_input_ids = unlabel_batch['input_ids'].to(device)
    unlabel_attention_mask = unlabel_batch['attention_mask'].to(device)
    text = unlabel_batch['text']

    # Forward Pass for detection task and compute its loss
    detection_logits = mymodel(label_input_ids, label_attention_mask, task='detection')
    detection_loss = criterion(detection_logits, label)   # CrossEntropyLoss in PyTorch expect raw logits as input
    
    if epoch == 1:
      masked_input_ids, mlm_labels = masked_tokens(unlabel_input_ids, tokenizer)
      mlm_logits = mymodel(masked_input_ids, unlabel_attention_mask, task='mlm')
      mlm_loss_raw = criterion(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
      # Filter out tokens with label -100 (they're not masked)
      masked_mask = mlm_labels.view(-1) != -100
      mlm_loss = (mlm_loss_raw * masked_mask).sum() / masked_mask.sum()   # mean loss over just the masked tokens
      # Gradually increase the weight for the auto-labeling loss
      #current_weight = initial_weight + (epoch-2) * weight_increment   # 0.01, 0.03, 0.05
      
      optimizer.zero_grad()
      total_batch_loss = detection_loss + (mlm_weight * mlm_loss)
      total_batch_loss.backward()
    
    else:
      optimizer.zero_grad()
      total_batch_loss = detection_loss
      total_batch_loss.backward()
      
    # Backward Pass and Optimization
    
    optimizer.step()
    torch.cuda.empty_cache()

    total_loss += total_batch_loss.item()

    # Calcuate accuracy
    predictions = torch.argmax(detection_logits, dim=1)   # same as logits.argmax(dim=1) Take bigger logit between two labels and output its index which is the label value.
    correct_prediction += (predictions == label).sum().item()   # item() converts tensor to python integer
    total_prediction += label.size(0)

    if epoch == num_epoch - 1:
      with torch.no_grad():
        autolabeling_logits = mymodel(unlabel_input_ids, unlabel_attention_mask, task='detection')  
        # Generate pseudo-labels
        score = torch.softmax(autolabeling_logits, dim=1) # Convert tensor of logits to probability over each label. Those return confidence score and each corresponding index will be labels (0 or 1)
        confidence_score, pseudo_label = torch.max(score, dim=1)
          
        confidence_score = confidence_score.cpu().tolist()
        pseudo_label = pseudo_label.cpu().tolist()

        # Stor pseudo labeled data for further training
        for text, label, score in zip(text, pseudo_label, confidence_score):
          pseudo_labeled_data.append({
            'text': text,
            'label': label,
            'score': round(score,3)})
    
    torch.cuda.empty_cache() 

  avg_loss = total_loss / len(train_loader)
  avg_accuracy = (correct_prediction / total_prediction) * 100
  print(f"[MTL] Epoch {epoch+1}/{num_epoch}, Avg Loss: {avg_loss:.2f}, Avg Accuracy: {avg_accuracy}")

  torch.cuda.empty_cache()

# Save the fine-tuned KoELECTRA model
model_save_path = "/proj/uppmax2024-2-24/hanna/koelectra_finetune_v1.pth"
torch.save(mymodel.state_dict(), model_save_path)

print(f"Model saved at {model_save_path}")

# Save psuedo labeled data as a file

def save_pseudo_labeled_data(filename, data):
  with open(filename, 'w', encoding='utf-8') as file:
    for item in data:
      file.write(f"{item['text']}\t{item['label']}\t{item['score']}\n")

pseudo_save_path = "/proj/uppmax2024-2-24/hanna/koelectra_pseudo_data.txt"
save_pseudo_labeled_data(pseudo_save_path, pseudo_labeled_data)
print(f"Pseudo-labeled data is saved at {pseudo_save_path}")

print("Ready to test")

# Test model

# label 1: positive sample (hate speech)
# label 2: negative sample (non-hate speech)
# True positive (Correctly predicted hate speech), False positive (Non-hate speech mistakenly predicted as hate speech)
# True negative (Correctly predicted non-hate speech), False negative (Hate speech mistakenly predicted as non-hate speech)

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

mymodel = MultiTaskModel(koelectra_model, num_label)
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

    outputs = mymodel(input_ids, attention_mask, task='detection')
    predictions = torch.argmax(outputs, dim=1)
    correct += (predictions == label).sum().item()
    total += label.size(0)
    for i in range(len(text)):
      result.append([text[i], predictions[i], label[i]])

  accuracy = (correct / total) * 100
  print(f"[Test] Accuracy: {accuracy:.3f}")

# Save result to a text file

test_save_path = "/proj/uppmax2024-2-24/hanna/koelectra_test_result_v1.txt"

with open(test_save_path, 'w', encoding='utf-8') as f_result:
    for data in result:
        text, prediction, label = data
        f_result.write(f"{text}\t{prediction}\t{label}\n")
        
print(f"Test result is saved at {test_save_path}")

# Calculate precision, recall, F1

all_predictions = [data[1].cpu().item() for data in result]  # Extract predictions from result
all_labels = [data[2].cpu().item() for data in result]  # Extract labels from result

precision = precision_score(all_labels, all_predictions)   # TP / TP+FP (how many predicted hate speech samples were actually hate speech)
recall = recall_score(all_labels, all_predictions)         # TP/ TP+FN (how many actual hate speech samples were correctly detected)
f1 = f1_score(all_labels, all_predictions)                 # 2 * (precision * recall) / (precision + recall)   (indicates a better balance between precision and recall)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

print("Precision, Recall, F1 Scores are done.")


file_path = "/proj/uppmax2024-2-24/hanna/koelectra_test_result_v1.txt"

preds = []
labels = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue  # skip malformed lines
        _, pred, label = parts
        preds.append(int(pred))
        labels.append(int(label))

# Optional: see label/prediction distribution
from collections import Counter
print("Prediction distribution:", Counter(preds))
print("Label distribution:", Counter(labels))

# Compute metrics
print("Classification Report:")
print(classification_report(labels, preds, digits=3))

