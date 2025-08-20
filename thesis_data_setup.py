# 1. Combine hate speech datapoints categorized under specific target groups from different sources
# 2. Consider distribution of datapoints over target groups
# 3. Consider distribution between hate speech and non-hate speech datapoints
# 4. Consider datapoints with multiple target group catetories
# 5. Consider setting common names of labels among different datasets
#    (Hate: gender, origin, politics, religion, physical, age, job, profanity), region, others / None-Hate: clear)

!pip install transformers
!pip install datasets==1.17.0

# UnSmile
from datasets import load_dataset
from collections import Counter
import re

dataset = load_dataset('smilegate-ai/kor_unsmile')

# Number of datapoints
for split in dataset:
  data_dict = dataset[split] # train or valid
  total_datapoint = len(data_dict['문장'])
  print(split, total_datapoint) # Train: 15 005, Valid: 3737, Total: 18 742


# Count datapoint with multi-labels (Multiple labels: checked with 1 in more than one label index in the value of label key)
multi_label_count = sum(sum(label) >=2 for split in dataset for label in dataset[split]['labels'])
print(f"Multi-labeled labels: {multi_label_count}")

single_label_count = sum(sum(label) == 1 for split in dataset for label in dataset[split]['labels'])
print(f"single-labeled count: {single_label_count}")

no_label_count = sum(sum(label) == 0 for split in dataset for label in dataset[split]['labels'])
print(f"No-Labeled Count: {no_label_count}")

# Label mapping
label_index = {0: "gender", 1: "gender", 2: "gender", 3: "origin", 4: "age", 5: "region", 6: "religion", 7: "others", 8: "profanity", 9: "clear" }

def data_setup(dataset):
  new_dict = {'text': [], 'label': []}
  for split in dataset:
    data_dict = dataset[split] # train or valid
    for text, label in zip(data_dict['문장'], data_dict['labels']):
      if sum(label) == 1:
        index = label.index(1)
        if label_index[index] == 'others': # Igonore others category because it doesn't belong to any major category
          continue
        new_dict['text'].append(text)
        new_dict['label'].append(label_index[index])
      elif sum(label) > 1:
        new_dict['text'].append(text)
        new_dict['label'].append('multi')

  return new_dict

unsmile_new_data = data_setup(dataset)

print(f"Number of text: {len(unsmile_new_data['text'])}") # (Train+Valid) - (Multi-labeled) - (No-labeled) = 18 742 - 1228 - 2 = 17 512
print(f"Number of label: {len(unsmile_new_data['label'])}")

# Number of each label
counter = Counter(unsmile_new_data['label'])
print(counter)

# KOLD

import json

with open('kold_v1.json', 'r') as file:
  dataset = json.load(file)
#print(dataset[100:300])

# Total number of datapoints: 40 429
print(f"Number of datadapoint : {len(dataset)}")

# Count hate speech and non-hate speech datapoints (hate speech datapoints: only 'group' and 'untargeted(will be categorized under profanity)' value in 'TGT' key): 35 128
count = 0
for data in dataset:
  if data['TGT'] == 'group' or data['TGT'] == 'untargeted' or data['TGT'] is None:
    count += 1
print(f"Number of hate speech and non-hate speech : {count}")


# Count datapoints with multi labels (Multi labels: including '&' in the value of'GRP' key) # 278
count = 0
for data in dataset:
  if data['TGT'] == 'group' and '&' in data['GRP']:
    count += 1
print(f"Number of datapoints with multi labels : {count}")


def data_setup(dataset):
  new_dict = {'text': [], 'label': []}
  for data in dataset:
    if data['TGT'] == 'group' and '&' not in data['GRP']:
      new_label_name = data['GRP'].split('-')[0]
      if new_label_name == 'race':
        new_label_name = 'origin'
      elif new_label_name == 'others':   # Igonore others category because it doesn't belong to any major category
        continue
      new_dict['text'].append(data['comment'])
      new_dict['label'].append(new_label_name)
    elif data['TGT'] == 'untargeted':
      new_dict['text'].append(data['comment'])
      new_dict['label'].append('profanity')
    elif data['TGT'] is None:
      new_dict['text'].append(data['comment'])
      new_dict['label'].append('clear')
    elif data['TGT'] == 'group' and '&' in data['GRP']:
      new_dict['text'].append(data['comment'])
      new_dict['label'].append('multi')
  return new_dict

kold_new_data = data_setup(dataset)

print(f"Number of text: {len(kold_new_data['text'])}") # 32 254
print(f"Number of label: {len(kold_new_data['label'])}") # 32 254

# Number of label
counter = Counter(kold_new_data['label'])
print(counter)   # hate speech datapoints/non hate speech data - datapoints with multi labels (35 128 - 278) = 34 850

# K-MHas

def load_data(file_path):
  with open(file_path, 'r') as file:
    dataset = file.read()
    dataset = dataset.split('\n')
    dataset = dataset[1:-1] # remove head of text (document, label) and empty line at the end of text file
    return dataset

label_map = {'0': 'origin', '1': 'physical', '2': 'politics', '3': 'profanity', '4': 'age', '5': 'gender', '6': 'origin', '7': 'religion', '8': 'clear'}

def data_setup(dataset):
  new_dict = {'text': [], 'label': []}
  for data in dataset:
    data = data.split('\t')
    if len(data) > 1 and ',' not in data[1] and data[1] in label_map:
      data[0] = data[0].replace('"', '').replace("'", "")
      new_dict['text'].append(data[0])
      new_dict['label'].append(label_map[data[1]])
    elif len(data) > 1 and ',' in data[1]:
      data[0] = data[0].replace('"', '').replace("'", "")
      new_dict['text'].append(data[0])
      new_dict['label'].append('multi')

  return new_dict

# load data
train = load_data('kmhas_train.txt')
valid = load_data('kmhas_valid.txt')
test = load_data('kmhas_test.txt')

# Total number of datapoints
print(f"Train: {len(train)}")
print(f"Valid: {len(valid)}")
print(f"Test: {len(test)}")

# Count datapoints with multiple labels
data_list = ['train', 'valid', 'test']
data_dict = {'train': train, 'valid': valid, 'test': test}
for l in data_list:
  count = 0
  for data in data_dict[l]:
    data = data.split('\t')
    if len(data) > 1 and ',' in data[1]:
      count += 1
  print(f"Multiple labels in {l}: {count}") # Train: 69 138, Valid: 7762

new_train_data = data_setup(train)
new_valid_data = data_setup(valid)
new_test_data = data_setup(test)

print(f"Total of train text: {len(new_train_data['text'])}", f"Total of train label: {len(new_train_data['label'])}") # 78 977 - 9839 (Total number of data - datapoints with multiple labels) = 69 138
print(f"Total of train text: {len(new_valid_data['text'])}", f"Total of train label: {len(new_valid_data['label'])}") # 8776 - 1014 (Total number of data - datapoints with multiple labels) = 7762
print(f"Total of train text: {len(new_test_data['text'])}", f"Total of train label: {len(new_test_data['label'])}") # 21 939 - 2754 (Total number of data - datapoints with multiple labels) = 19185


kmhas_new_data = {
    key: new_train_data[key] + new_valid_data[key] + new_test_data[key]
    for key in ['text', 'label']}

print(f"Total of text from train, valid, test: {len(kmhas_new_data['text'])}")
print(f"Total of label from train, valid, test: {len(kmhas_new_data['label'])}") # 96 085


# Number of label
counter = Counter(kmhas_new_data['label'])
print(counter)

# K-HATERS
import json
import random
from collections import Counter
import re

data_list = ['train.jsonl', 'val.jsonl', 'test.jsonl']
dataset = []
def data_load(file_name):
  for name in file_name:
    with open(name, 'r') as file:
      for line in file:
        dataset.append(json.loads(line.strip()))

# Total number of datapoints
data_load(data_list)
print(len(dataset))   # 192 158

# Number of datapoints with multiple labels
single_dataset = []   # Contain single target group data points and clear data points.
multi_dataset = []
for data in dataset:
  label_keys = list(data.keys())[1:10]
  count = sum(1 for key in label_keys if data[key] in {1, 2})
  if count <= 1:
    single_dataset.append(data)
  else:
    multi_dataset.append(data)

print(f"Number of single data: {len(single_dataset)}") # 176 721
print(f"Number of multi data: {len(multi_dataset)}")   # 15 437         # 176 721 + 15 437 = 192 158

# Count clear datapoints
c = 0
for data in single_dataset:
  if all(data[key] == 0 for key in list(data.keys())[1:14]):
    c += 1
print(f"Clear data points: {c}")   # 52 061

# Data setup
def data_setup(single_dataset, multi_dataset):
  new_dict_with_placeholder = {'text': [], 'label': []}
  new_dict_without_placeholder = {'text': [], 'label': []}

  for data in single_dataset:
    label_keys= list(data.keys())[1:10]   # Filter out data points categorized 'individuals' and 'others'
    #label_out_list = {'individuals', 'insult', 'swear', 'threat','obscenity'}
    label = [key for key in label_keys if data[key] in {1, 2}]
    #print(label)

    # Determine the label for each data point
    if all(data[key] == 0 for key in list(data.keys())[1:14]):
      label = 'clear'
    elif label:
      label = label[0]
      if label == 'individuals' or label == 'others':
        continue
      elif label == 'race':
        label = 'origin'
      elif label == 'disability':
        label = 'physical'
    if not label:
      continue


    # Process text
    text = data['text']

    # Check if the text contains the placeholders
    count1 = text.count('#@이름#')
    count2 = text.count('OOO')

    # Add the data point to the appropriate dataset
    if count1 >= 1 or count2 >= 1:
      new_dict_with_placeholder['text'].append(text)
      new_dict_with_placeholder['label'].append(label)
    else:
      new_dict_without_placeholder['text'].append(text)
      new_dict_without_placeholder['label'].append(label)


  for data in multi_dataset:
    text = data['text']
    label = 'multi'

    # Check if the text contains the placeholders
    count1 = text.count('#@이름#')
    count2 = text.count('OOO')

    # Add the data point to the appropriate dataset
    if count1 >= 1 or count2 >= 1:
      new_dict_with_placeholder['text'].append(text)
      new_dict_with_placeholder['label'].append(label)
    else:
      new_dict_without_placeholder['text'].append(text)
      new_dict_without_placeholder['label'].append(label)

  # Return both datasets
  return new_dict_with_placeholder, new_dict_without_placeholder


khaters_new_data_with_placeholder, khaters_new_data_without_placeholder  = data_setup(single_dataset, multi_dataset)

print(f"Number of text without p: {len(khaters_new_data_without_placeholder['text'])}")
print(f"Number of label without p: {len(khaters_new_data_without_placeholder['label'])}")

print(f"Number of text with p: {len(khaters_new_data_with_placeholder['text'])}")
print(f"Number of label with p: {len(khaters_new_data_with_placeholder['label'])}")

print(khaters_new_data_with_placeholder['text'][:10])
print(khaters_new_data_with_placeholder['label'][:10])
# Number of label
counter_without = Counter(khaters_new_data_without_placeholder['label'])
counter_with = Counter(khaters_new_data_with_placeholder['label'])   # Hate: 17307, None-hate: 11744 -> Size of hate speech data point need to be reducde to 11744. (Will reduce politics, multi size)
print(counter_without)
print(counter_with)

# Reduce data size of politics, multi from hate speech data points
# Politics from 7646 to 4864, multi from 6898 to 4117

# Separate dataset by labels
politics_data = {'text': [], 'label':[]}
multi_data = {'text': [], 'label':[]}
other_data = {'text': [], 'label':[]}
for text, label in zip(khaters_new_data_with_placeholder['text'], khaters_new_data_with_placeholder['label']):
  if label == 'politics':
    politics_data['text'].append(text)
    politics_data['label'].append(label)
  elif label == 'multi':
    multi_data['text'].append(text)
    multi_data['label'].append(label)
  else:
    other_data['text'].append(text)
    other_data['label'].append(label)

# Reduce the dataset size for "politics" and "multi"
politics_indices = random.sample(range(len(politics_data['text'])), 4864)
multi_indices = random.sample(range(len(multi_data['text'])), 4117)

# Create reduced datasets
politics_data_reduced = {
    'text': [politics_data['text'][i] for i in politics_indices],
    'label': [politics_data['label'][i] for i in politics_indices]
}

multi_data_reduced = {
    'text': [multi_data['text'][i] for i in multi_indices],
    'label': [multi_data['label'][i] for i in multi_indices]
}

# Merge back into a final dataset
final_dataset = {
    'text': politics_data_reduced['text'] + multi_data_reduced['text'] + other_data['text'],
    'label': politics_data_reduced['label'] + multi_data_reduced['label'] + other_data['label']
}

print(len(final_dataset['text']))
print(len(final_dataset['label']))

print(Counter(final_dataset['label']))

# Save or return processed dataset
print(final_dataset['text'][:5])  # Print first 5 entries as a sample
print(final_dataset['label'][:5])

# Replace #@이름#, OOO with real names

import itertools

real_names = [
    "서준", "민준", "도윤", "예준", "하준", "시우", "지호", "주원", "지후", "도현",
    "준우", "준서", "건우", "우진", "현우", "선우", "지훈", "은우", "유준", "연우",
    "서진", "이준", "시윤", "민재", "현준", "정우", "윤우", "수호", "승우", "지우",
    "유찬", "지환", "승현", "준혁", "시후", "승민", "이안", "진우", "민성", "수현",
    "지원", "준영", "시현", "한결", "재윤", "지한", "우주", "지안", "태윤", "은호",
    "서윤", "서연", "지우", "하윤", "서현", "하은", "민서", "지유", "윤서", "지아",
    "채원", "수아", "지민", "서아", "지윤", "다은", "지안", "은서", "하린", "소율",
    "예은", "예린", "수빈", "소윤", "유나", "예원", "지원", "시은", "채은", "아린",
    "윤아", "시아", "유진", "예나", "아윤", "예서", "가은", "유주", "하율", "연우",
    "민지", "예진", "주아", "서영", "다인", "서우", "나은", "수연", "연서", "수민"
]

# Separate hate and non-hate speech
hate_speech = {'text': [], 'label': []}
non_hate_speech = {'text': [], 'label': []}

for text, label in zip(final_dataset['text'], final_dataset['label']):
    if label == "clear":
        non_hate_speech['text'].append(text)
        non_hate_speech['label'].append(label)
    else:
        hate_speech['text'].append(text)
        hate_speech['label'].append(label)

# Ensure dataset sizes match expectations
assert len(hate_speech['text']) == 11744, "Hate speech count mismatch!"
assert len(non_hate_speech['text']) == 11744, "Non-hate speech count mismatch!"

# Assign names evenly
num_names = len(real_names)  # 100 names
base_count = len(hate_speech['text']) // num_names  # 117 per name
extra_count = len(hate_speech['text']) % num_names  # 44 extra samples

# Shuffle names for fairness
random.shuffle(real_names)

# Function to replace placeholders
def replace_name(text, name):
    return text.replace("#@이름#", name).replace("OOO", name)

# Distribute names evenly
name_iterator = itertools.cycle(real_names)  # Cycle names

for i in range(11700):  # First 11700 (117 * 100) assignments
    assigned_name = next(name_iterator)
    hate_speech['text'][i] = replace_name(hate_speech['text'][i], assigned_name)
    non_hate_speech['text'][i] = replace_name(non_hate_speech['text'][i], assigned_name)

# Handle remaining 44 hate and 44 non-hate speech samples
for i in range(11700, 11744):
    assigned_name = real_names[i % num_names]  # Use first 44 names again
    hate_speech['text'][i] = replace_name(hate_speech['text'][i], assigned_name)
    non_hate_speech['text'][i] = replace_name(non_hate_speech['text'][i], assigned_name)

# Merge back into final dataset
final_dataset = {
    'text': hate_speech['text'] + non_hate_speech['text'],
    'label': hate_speech['label'] + non_hate_speech['label']
}

# Print sample results
print(final_dataset['text'][:5])  # Print first few samples
print(final_dataset['label'][:5])  # Print first few labels

# Combine haters_new_data_without_placeholder and final_dataset (adjusted version of khaters_new_data_with_placeholder)

# Merge the two datasets
khaters_new_data = {
    'text': final_dataset['text'] + khaters_new_data_without_placeholder['text'],
    'label': final_dataset['label'] + khaters_new_data_without_placeholder['label']
}

print(len(khaters_new_data['text']))
print(len(khaters_new_data['label']))

print(Counter(khaters_new_data['label']))

# Data cleaning
# 1. Remove sentences containing less than 10 korean characters
# 2. Reduce repeated characters more then three to two
# 3. Remain one sentence if exactly same sentences are found
# 4. Remove sentences contain only numbers, only Jamo charaters or Emoji characters

import unicodedata
import re

def clean_data(noisy_data):
  cleaned_texts, cleaned_labels = [], []
  for text, label in zip(noisy_data['text'], noisy_data['label']):
    # Reduce repeated syllables more than three to two
    text = re.sub(r'([\uAC00-\uD7A3])\1{2,}', r'\1\1', text)
    text = re.sub(r'([.,:;!?~=#%&/()])\1{2,}', r'\1\1', text)
    # Normalize to NFD (decomposes Hangul syllables into Jamo)
    decomposed_text = unicodedata.normalize("NFD", text)

    # Reduce repeated characters more then three to two
    cleaned_text = re.sub(r'([\u1100-\u11FF\u3130-\u318F])\1{2,}', r'\1\1', decomposed_text)

    # Match all Jamo (initial, medial, final) + Compatibility Jamo
    jamo_pattern = re.compile(r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]')
    matched_jamo = jamo_pattern.findall(cleaned_text)

    jamo_len = len(matched_jamo)
    only_num = re.compile(r'^\d+$')

    if (only_num.match(cleaned_text) or jamo_len < 5):
      continue
    cleaned_texts.append(unicodedata.normalize("NFC", cleaned_text))
    cleaned_labels.append(label)

  return {'text': cleaned_texts, 'label': cleaned_labels}


cleaned_unsmile_data = clean_data(unsmile_new_data)
#print(cleaned_unsmile_data)

cleaned_kold_data = clean_data(kold_new_data)
#print(cleaned_kold_data)

cleaned_kmhas_data = clean_data(kmhas_new_data)
#print(cleaned_kmhas_data)

cleaned_khaters_data = clean_data(khaters_new_data)
#print(cleaned_khaters_data)

print(f"Number of text from unsmile: {len(cleaned_unsmile_data['text'])}")
print(f"Number of label from unsmile: {len(cleaned_unsmile_data['label'])}")

print(f"Number of text from kold: {len(cleaned_kold_data['text'])}")
print(f"Number of label from kold: {len(cleaned_kold_data['label'])}")

print(f"Number of text from K-Mhas: {len(cleaned_kmhas_data['text'])}")
print(f"Number of label from K-Mhas: {len(cleaned_kmhas_data['label'])}")

print(f"Number of text from K-Hates: {len(cleaned_khaters_data['text'])}")
print(f"Number of label from K-Hates: {len(cleaned_khaters_data['label'])}")

# Number of label

print(Counter(cleaned_unsmile_data['label']))
print(Counter(cleaned_kold_data['label']))
print(Counter(cleaned_kmhas_data['label']))
print(Counter(cleaned_khaters_data['label']))


# Combine all cleaned dataset

all_data = {key: cleaned_unsmile_data[key] + cleaned_kold_data[key] + cleaned_kmhas_data[key] + cleaned_khaters_data[key]
    for key in ['text', 'label']}

print(f"Number of text: {len(all_data['text'])}, Number of label: {len(all_data['label'])}")
print(all_data['text'][:10])
print(all_data['label'][:10])

print(all_data['text'][-1:-10:-1])
print(all_data['label'][-1:-10:-1])

# Number of label
counter = Counter(all_data['label'])
print(counter)

from collections import Counter

counter = Counter(all_data['text'])
print(f"Total unique text: {len(counter)}")
duplicate = []
unique = []

for i in counter:
  if counter[i] > 1:
    duplicate.append(i)
  elif counter[i] == 1:
    unique.append(i)

print(f"Duplicated text: {len(duplicate)}")
print(f"Unique text: {len(unique)}")

# Duplicated text check in all_data

def remove_duplicate(data):
  seen = set()
  unique_text = []
  unique_label = []
  for text, label in zip(data['text'], data['label']):
    if text not in seen:
      seen.add(text)
      unique_text.append(text)
      unique_label.append(label)

  return {'text': unique_text, 'label': unique_label}

unique_new_label = remove_duplicate(all_data)

print(f"Number of text: {len(unique_new_label['text'])}, Number of label: {len(unique_new_label['label'])}")

from collections import Counter

counter = Counter(unique_new_label['text'])
print(f"Total unique text: {len(counter)}")
duplicate = []
unique = []

for i in counter:
  if counter[i] > 1:
    duplicate.append(i)
  elif counter[i] == 1:
    unique.append(i)

print(f"Duplicated text: {len(duplicate)}")
print(f"Unique text: {len(unique)}")

# Unlabeled data
# Extract data from HateSCore.csv, apeach.test.csv, BEEP! unlabeled datasets (5 files)
# Data clean

import unicodedata
import re
import random

def clean_unlabel(filename):
  with open(filename, 'r', encoding='utf-8') as file:
    texts = file.read()
    texts = texts.split("\n")
    texts = random.sample(texts, min(100000, len(texts)))
    print(f"Number of {filename} sampled text: {len(texts)}")

  cleaned_texts = []
  for text in texts:
    # Reduce repeated syllabels more then three to two
    text = re.sub(r'([\uAC00-\uD7A3])\1{2,}', r'\1\1', text)
    text = re.sub(r'([.,!?~=#%&/();:])\1{2,}', r'\1\1', text)
    # Normalize to NFD (decomposes Hangul syllables into Jamo)
    decomposed_text = unicodedata.normalize("NFD", text)

    # Reduce repeated characters more then three to two
    cleaned_text = re.sub(r'([\u1100-\u11FF\u3130-\u318F])\1{2,}', r'\1\1', decomposed_text)

    # Match all Jamo (initial, medial, final) + Compatibility Jamo
    jamo_pattern = re.compile(r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]')
    matched_jamo = jamo_pattern.findall(cleaned_text)

    jamo_len = len(matched_jamo)
    only_num = re.compile(r'^\d+$')

    if (only_num.match(cleaned_text) or jamo_len < 5):
      continue
    cleaned_texts.append(unicodedata.normalize("NFC", cleaned_text))

  return cleaned_texts

# List of files
files = [
    "unlabeled_comments_1.txt",
    "unlabeled_comments_2.txt",
    "unlabeled_comments_3.txt",
    "unlabeled_comments_4.txt",
    "unlabeled_comments_5.txt"
]

# Process and sample from each
beep_cleaned = []
hatescore_cleaned = []
apeach_cleaned = []

for file in files:
  cleaned = clean_unlabel(file)
  print(f"Number of {file} cleaned text: {len(cleaned)}")
  beep_cleaned.extend(cleaned)

print(f"Number of BEEP cleaned text: {len(beep_cleaned)}")

hatescore = clean_unlabel("hatescore.txt")
hatescore_cleaned.extend(hatescore)
print(f"Number of Hatesocre cleaned text: {len(hatescore)}")

apeach = clean_unlabel("apeach.txt")
apeach_cleaned.extend(apeach)
print(f"Number of Apeach cleaned text: {len(apeach)}")

print(f"Total number of text: {len(beep_cleaned)+len(hatescore_cleaned)+len(apeach_cleaned)}")

# Duplicated text check in BEEP unlabeled data

def beep_remove_duplicate_unlabel(beep_cleaned):
  seen = set()
  unique_text = []
  dup_text = []
  for text in beep_cleaned:
    if text not in seen:
      seen.add(text)
      unique_text.append(text)
    elif text in seen:
      dup_text.append(text)
  print(f"numbe of duplicated text: {len(dup_text)}")

  return unique_text

beep_unique_unlabel = beep_remove_duplicate_unlabel(beep_cleaned)

print(f"Number of unique text: {len(beep_unique_unlabel)}")   # 445292 - 18131 = 427161

# BEEP
beep_duplicated_text_label_unlabel = [text for text in unique_new_label['text'] if text in beep_unique_unlabel]

print(f"The number of duplicated between label and unlabeled data: {len(beep_duplicated_text_label_unlabel)}")   # 15123

# HateScore
hatescore_duplicated_text_label_unlabel = [text for text in unique_new_label['text'] if text in hatescore_cleaned]

print(f"The number of duplicated between label and unlabeled data: {len(hatescore_duplicated_text_label_unlabel)}")   # 15123

# APEACH
apeach_duplicated_text_label_unlabel = [text for text in unique_new_label['text'] if text in apeach_cleaned]

print(f"The number of duplicated between label and unlabeled data: {len(apeach_duplicated_text_label_unlabel)}")   # 15123

# Remove duplicated texts from unlabel
def remove_duplicate_between_label_unlabel(beep_duplicated_text_label_unlabel):
  duplicated_set = set(beep_duplicated_text_label_unlabel)
  print(len(duplicated_set))
  return [text for text in beep_unique_unlabel if text not in duplicated_set]

beep_unique = remove_duplicate_between_label_unlabel(beep_duplicated_text_label_unlabel)

print(f"Number of unique new unlabel text: {len(beep_unique)}")   # 412374 - 15174 = 397 200

# Remove duplicated texts from unlabel
def remove_duplicate_between_label_unlabel(hatescore_duplicated_text_label_unlabel):
  duplicated_set = set(hatescore_duplicated_text_label_unlabel)
  print(len(duplicated_set))
  return [text for text in hatescore_cleaned if text not in duplicated_set]

hatescore_unique = remove_duplicate_between_label_unlabel(hatescore_duplicated_text_label_unlabel)

print(f"Number of unique new unlabel text: {len(hatescore_unique)}")   # 11105 - 2 = 111103

# Remove duplicated texts from unlabel
def remove_duplicate_between_label_unlabel(apeach_duplicated_text_label_unlabel):
  duplicated_set = set(apeach_duplicated_text_label_unlabel)
  print(len(duplicated_set))
  return [text for text in apeach_cleaned if text not in duplicated_set]

apeach_unique = remove_duplicate_between_label_unlabel(apeach_duplicated_text_label_unlabel)

print(f"Number of unique new unlabel text: {len(apeach_unique)}")   # 3770 - 1 = 3769

import random

# Assume beep_unique is already defined as per your previous code

# Set the desired number of random samples
sample_size = 185979

# Ensure you don't request more samples than available items
if sample_size > len(beep_unique):
    raise ValueError("Sample size exceeds the number of available unique texts.")

# Randomly sample the texts
beep_sampled = random.sample(beep_unique, sample_size)

# Optional: print sample size to confirm
print(f"Number of randomly sampled texts: {len(beep_sampled)}")

# Combine the three lists into one
unique_combined_unlabel = beep_sampled + hatescore_unique + apeach_unique

# Optional: print total number of combined texts
print(f"Total number of combined texts: {len(unique_combined_unlabel)}")

# Save unlabeled unique data

def save_unlabel_data(filename, data):
  with open(filename, 'w', encoding='utf-8') as file:
    for text in data:
      file.write(f"{text}\n")
    print(f"Data saved to {filename}")

save_unlabel_data('unlabel_data.txt', unique_combined_unlabel)

# Distribution: Length of datapoints based on syllabels
def count_korean_syllabel(text):
    korean_pattern = re.compile(r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]')
    return len(korean_pattern.findall(text))

def text_len(new_data):
  len_dict = {'1-10': 0, '11-20': 0, '21-40': 0, '40+' : 0}
  new_data = new_data['text']
  for data in new_data:
    data_len = count_korean_syllabel(data)
    if data_len <= 10:
      len_dict['1-10'] += 1
    elif data_len > 10 and data_len <= 20:
      len_dict['11-20'] += 1
    elif data_len > 20 and data_len <= 40:
      len_dict['21-40'] += 1
    else:
      len_dict['40+'] += 1
  return len_dict

print(text_len(unique_new_data))

print(unique_new_data['label'][:10])
print(unique_new_data['label'][-10:])

# Convert labels to binary form (1 for anything not 'clean', 0 for 'clean')
unique_new_data['label'] = [1 if label != 'clear' else 0 for label in unique_new_data['label']]

# Print first 10 and last 10 labels
print(unique_new_data['label'][:10])
print(unique_new_data['label'][-10:])

print(len(unique_new_data['label']))
counter = Counter(unique_new_data['label'])
print(counter)

# Divide data to subset: Train, Test
import random

hate_data = [(text, label) for text, label in zip(unique_new_data['text'], unique_new_data['label']) if label == 1]
clear_data = [(text, label) for text, label in zip(unique_new_data['text'], unique_new_data['label']) if label == 0]

# Shuffle data points in Hate and Clear
random.shuffle(hate_data)
random.shuffle(clear_data)

train_hate, test_hate = 99365, 33121   #92740, 6624, 33122
train_clear, test_clear = 101486, 33828   #94721, 6766, 33829

train_data = hate_data[:train_hate] + clear_data[:train_clear]
test_data = hate_data[train_hate:train_hate+test_hate] + clear_data[train_clear:train_clear+test_clear]

print(len(train_data))
print(len(test_data))

# Shuffle data points in each subset
random.shuffle(train_data)
random.shuffle(test_data)

# Convert it to dic

train_data = {'text': [text for text, label in train_data], 'label': [label for text, label in train_data]}
test_data = {'text': [text for text, label in test_data], 'label': [label for text, label in test_data]}

# Print final sizes to confirm correct splitting
print(f"Train Set: Hate= {train_data['label'].count(1)}, Clean= {train_data['label'].count(0)}")
print(f"Test Set: Hate= {test_data['label'].count(1)}, Clean= {test_data['label'].count(0)}")

print(f"Total number of data: {len(train_data['text'])+ len(test_data['text'])}")


# Save unique new data

def save_new_data(filename, data):
  with open(filename, 'w', encoding='utf-8') as file:
    for text, label in zip(data['text'], data['label']):
      file.write(f"{text}\t{label}\n")
    print(f"Data saved to {filename}")

save_new_data('train_data.txt', train_data)
save_new_data('test_data.txt', test_data)

# Remove URL from unlabeled data

import re

with open('old_unlabel_data.txt', 'r', encoding='utf-8') as file:
  data = file.read()
  data = data.split("\n")


def remove_urls(text):
    return re.sub(r"http[s]?://\S+", "", text).strip()

cleaned_data = [remove_urls(text) for text in data]

with open('unlabel_data.txt', 'w', encoding='utf-8') as file:
    for text in cleaned_data:
        file.write(f"{text}\n")
    print(f"Data saved to unlabel_data.txt")

import random

# Set file paths
input_file_path = "unlabel_data.txt"
output_file_path = "sampled_unlabel_data.txt"

# Define number of samples to extract
num_samples = 401702

# Read all lines from file
with open(input_file_path, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

# Sanity check
print(f"Total lines in original file: {len(all_lines)}")

# Randomly sample 200,851 lines
sampled_lines = random.sample(all_lines, num_samples)

# Write sampled lines to new file
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.writelines(sampled_lines)

print(f"Saved {len(sampled_lines)} lines to {output_file_path}")
