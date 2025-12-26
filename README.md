# Master thesis
- Language Technology  Master's program, Uppsala University
- Designing Korean hate speech detection models with Auto-labeling using Semi-supervised Multi-task learning: Sustainable approach

## Keywords
\# Hate speech detection # Semi-supervised learning # Multi-task learning # Masked langauge modeling
\# Auto-labeling # Synthetic data # Data augmentation # Sustainability

## Introduction
- This thesis proposes a sustainable alternative to supervised learning for Korean hate speech detection using semi-supervised multi-task learning.

## Data
- Korean hate/non-hate speech texts (Combined from different sources)
- Training: Labeled + Unlabeled
- Test: Labeled

<img width="400" height="200" alt="data" src="https://github.com/user-attachments/assets/160b804d-110f-4235-a255-1743ef9b5fc8" />

## Models
- Pre-trained KoBERT (https://github.com/SKTBrain/KoBERT), KoELECTRA (https://github.com/monologg/KoELECTRA)

## Evaluation
- For detection: Precision, recall, F1
- For auto-labeling: Confidence score, manual evaluation for samples by the author

## Experiments
- Data setup: thesis_data_setup.py
- First fine-tuning: thesis_kobert_finetune.py, thesis_koelectra_finetune.py
- Second fine-tuning: thesis_kobert_finetune_second.py, thesis_koelectra_finetune_second.py
- Generate synthetic datasets: thesis_pseudo_data_synthetic_data.py

<img width="550" height="450" alt="experimental workflow" src="https://github.com/user-attachments/assets/7f53f18c-47a8-4b56-8dc1-8f2d178fbb0c" />

## Results

<img width="800" height="500" alt="results" src="https://github.com/user-attachments/assets/c88c5202-3293-4134-beeb-7776074c095b" />

## Short conclusions
- The experimental results demonstrated that the proposed semi-supervised multi-task learning framework improved the performance of both KoBERT and KoELECTRA for the detection task.
- In addition, the models successfully generated useful pseudo-labeled data for the auto-labeling task, which in turn contributed to data augmentation and produced consistent, stable results overall.
- It raises important considerations regarding cost-effectiveness and sustainability in machine learning and artificial intelligence research.
