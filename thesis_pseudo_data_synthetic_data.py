# Filtering 1: Show Distribution of four confidence score buckets for each label

from collections import defaultdict

def filter_pseudo_data(file_path):
  label_count = defaultdict(int)

  buckets = {0: [0,0,0,0], 1: [0,0,0,0]}
  bucket_names = ["0–50", "51–70", "71–90", "91–100"]

  with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.read().strip().split('\n')   # Remove trailing newlines
    for line in lines:
      parts = line.split('\t')   # each text anc its label get splitted and saved as a string.

      label = int(parts[1])
      score = float(parts[2])

      label_count[label] += 1

      if score <= 0.5:
        buckets[label][0] += 1
      elif score <= 0.7:
        buckets[label][1] += 1
      elif score <= 0.9:
        buckets[label][2] += 1
      else:
        buckets[label][3] += 1

  print("Label Counts:")
  print(f"Total: {len(lines)}")
  print(f"Label 0: {label_count[0]}")
  print(f"Label 1: {label_count[1]}\n")

  print("Confidence Score Distribution by Bucket:")
  for label in [0, 1]:
    print(f"Label {label}:")
    for i, count in enumerate(buckets[label]):
      print(f"  {bucket_names[i]}: {count}")

print("KoBERT pseudo Data")
filter_pseudo_data('/content/kobert_pseudo_data.txt')

print("\nKoELECTRA pseudo Data")
filter_pseudo_data('koelectra_pseudo_data.txt')

# Sampling per confidence score range from the pseudo data

import random
from collections import defaultdict

def extract_samples_by_confidence(file_path):
    # Buckets: label -> list of samples
    buckets = {
        '51-70': {0: [], 1: []},
        '71-90': {0: [], 1: []},
        '91-100': {0: [], 1: []},
    }

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split('\n')

    for line in lines:
        parts = line.split('\t')
        sentence = parts[0]
        label = int(parts[1])
        score = float(parts[2])

        if 0.5 < score <= 0.7:
            buckets['51-70'][label].append((sentence, label))
        elif 0.7 < score <= 0.9:
            buckets['71-90'][label].append((sentence, label))
        elif 0.9 < score <= 1.0:
            buckets['91-100'][label].append((sentence, label))

    return buckets


def save_balanced_sample(buckets, bucket_name, num_per_label, output_prefix):
    sampled = []

    for label in [0, 1]:
        samples = buckets[bucket_name][label]
        if len(samples) < num_per_label:
            raise ValueError(f"Not enough samples in bucket {bucket_name} for label {label}")
        sampled += random.sample(samples, num_per_label)

    random.shuffle(sampled)

    output_file = f"{output_prefix}_{bucket_name.replace('-', '')}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, label in sampled:
            f.write(f"{sentence}\t{label}\n")

    print(f"Saved {len(sampled)} samples to {output_file}")


def save_combined_sample(buckets, output_prefix):
    config = {
        '51-70': 8424,
        '71-90': 8424,
        '91-100': 8424,
    }

    combined = []

    for bucket_name, num_per_label in config.items():
        for label in [0, 1]:
            samples = buckets[bucket_name][label]
            if len(samples) < num_per_label:
                raise ValueError(f"Not enough samples in bucket {bucket_name} for label {label}")
            combined += random.sample(samples, num_per_label)

    random.shuffle(combined)

    output_file = f"{output_prefix}_combined.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, label in combined:
            f.write(f"{sentence}\t{label}\n")

    print(f"Saved {len(combined)} combined samples to {output_file}")


def process_pseudo_data(file_path, output_prefix):
    buckets = extract_samples_by_confidence(file_path)

    # 1. 8424 samples from 51-70 bucket
    save_balanced_sample(buckets, '51-70', 8424, output_prefix)

    # 2. 8831 samples from 71-90 bucket
    save_balanced_sample(buckets, '71-90', 8424, output_prefix)

    # 3. 12885 samples from 91-100 bucket
    save_balanced_sample(buckets, '91-100', 8424, output_prefix)

    # 4. Combined extraction from all three buckets
    save_combined_sample(buckets, output_prefix)


# Run for KoBERT
print("Processing KoBERT Pseudo Data...")
process_pseudo_data('/content/kobert_pseudo_data.txt', 'kobert_filtered')

# Run for KoELECTRA
print("Processing KoELECTRA Pseudo Data...")
process_pseudo_data('koelectra_pseudo_data.txt', 'koelectra_filtered')

import os

def combine_datasets(original_file, pseudo_file, output_file):
    with open(original_file, 'r', encoding='utf-8') as orig_f:
        original_lines = orig_f.read().strip().split('\n')

    with open(pseudo_file, 'r', encoding='utf-8') as pseudo_f:
        pseudo_lines = pseudo_f.read().strip().split('\n')

    combined = original_lines + pseudo_lines

    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('\n'.join(combined))

    print(f"Created: {output_file} (Total: {len(combined)} lines)")


# Base paths
original_train_file = 'train_data.txt'

kobert_files = [
    'kobert_filtered_5170.txt',
    'kobert_filtered_7190.txt',
    'kobert_filtered_91100.txt',
    'kobert_filtered_combined.txt'
]

koelectra_files = [
    'koelectra_filtered_5170.txt',
    'koelectra_filtered_7190.txt',
    'koelectra_filtered_91100.txt',
    'koelectra_filtered_combined.txt'
]

# Combine and save for each pseudo dataset
for pseudo_file in kobert_files + koelectra_files:
    model_type = 'kobert' if 'kobert' in pseudo_file else 'koelectra'
    bucket = pseudo_file.split('_')[-1].replace('.txt', '')
    output_file = f'synthetic_data_{model_type}_{bucket}.txt'
    combine_datasets(original_train_file, pseudo_file, output_file)

# Manual sample check: Random samples of each confidence score bucket for each label for qualitative analysis

import random
from collections import defaultdict

def bucket_index(score):
    if score <= 0.5:
        return 0
    elif score <= 0.7:
        return 1
    elif score <= 0.9:
        return 2
    else:
        return 3

def bucket_name(index):
    return ["0-50", "51-70", "71-90", "91-100"][index]

def sample_pseudo_data(file_path, output_prefix, samples_per_bucket=100):
    samples_by_label_bucket = {0: defaultdict(list), 1: defaultdict(list)}

    # Read and bucket the data
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split('\n')
        for line in lines:
            parts = line.split('\t')
            label = int(parts[1])
            score = float(parts[2])
            bucket = bucket_index(score)
            samples_by_label_bucket[label][bucket].append(line)

    # Sample and write output
    for label in [0, 1]:
        selected = []
        print(f"\nLabel {label}:")
        for bucket in range(4):
            bucket_samples = samples_by_label_bucket[label][bucket]
            count = min(len(bucket_samples), samples_per_bucket)
            sampled = random.sample(bucket_samples, count) if count > 0 else []
            print(f"  Bucket {bucket} (Total: {len(bucket_samples)}): Sampled {len(sampled)}")

            output_file = f"{output_prefix}_{label}_{bucket_name(bucket)}.txt"
            with open(output_file, 'w', encoding='utf-8') as out:
              out.write('\n'.join(sampled))

# Example usage:
print("KoBERT")
sample_pseudo_data('kobert_pseudo_data.txt', output_prefix="kobert")

print("KoELECTRA")
sample_pseudo_data('koelectra_pseudo_data.txt', output_prefix="koelectra")

# Calculate accuracy between predictions and human annotation

def calculate_accuracy(file_path):
  total = 0
  correct = 0

  with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
      parts = line.strip().split('\t')
      #print(parts)
      prediction = int(parts[1])
      ground_truth = int(parts[3])
      total += 1

      if prediction == ground_truth:
        correct += 1

    if total == 0:
        print(f"No valid data to evaluate in '{file_path}'.")
        return 0.0

    accuracy = correct / total
    print(f"File: {file_path}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}\n")

# Example usage:
calculate_accuracy('kobert_0_0-50.txt')
calculate_accuracy('kobert_0_51-70.txt')
calculate_accuracy('kobert_0_71-90.txt')
calculate_accuracy('kobert_0_91-100.txt')

calculate_accuracy('kobert_1_0-50.txt')
calculate_accuracy('kobert_1_51-70.txt')
calculate_accuracy('kobert_1_71-90.txt')
calculate_accuracy('kobert_1_91-100.txt')

calculate_accuracy('koelectra_0_0-50.txt')
calculate_accuracy('koelectra_0_51-70.txt')
calculate_accuracy('koelectra_0_71-90.txt')
calculate_accuracy('koelectra_0_91-100.txt')

calculate_accuracy('koelectra_1_0-50.txt')
calculate_accuracy('koelectra_1_51-70.txt')
calculate_accuracy('koelectra_1_71-90.txt')
calculate_accuracy('koelectra_1_91-100.txt')
