"""
Self-training을 통한 데이터 relabeling 스크립트

이 스크립트는 다음 단계를 수행합니다:
1. 원본 데이터로 모델을 학습합니다.
2. 학습된 모델을 사용하여 증강된 데이터를 relabeling합니다.

주요 기능:
- BERT 기반 분류 모델 학습
- 학습된 모델을 사용한 데이터 relabeling
- 결과를 CSV 파일로 저장
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split

# 재현성을 위한 시드 설정
SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, 'Self_train_step3'),
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    report_to="wandb",
    logging_strategy='steps',
    eval_strategy='steps',
    save_strategy='steps',
    logging_steps=100,
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    learning_rate=2e-05,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    weight_decay=0.01,
    lr_scheduler_type='linear',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    greater_is_better=True,
    seed=SEED
)

# 모델 및 토크나이저 로드
use_checkpoint = False
model_path = os.path.join(OUTPUT_DIR, "self_train_step1/checkpoint-350")
model_name = model_path if use_checkpoint else "klue/roberta-large"

model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.inputs = []
        self.labels = []
        for text, label in zip(data['text'], data['target']):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# 데이터셋 로드 및 전처리
original_path = os.path.join(DATA_DIR, 'train.csv')
original_df = pd.read_csv(original_path)

dataset_train, dataset_valid = train_test_split(original_df, test_size=0.2, random_state=SEED)
data_train = BERTDataset(dataset_train, tokenizer)
data_valid = BERTDataset(dataset_valid, tokenizer)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 평가 메트릭 정의
f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')

# 트레이너 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# 학습된 모델을 사용하여 증강된 데이터 relabeling
augmented_path = os.path.join(DATA_DIR, 'augmented.csv')
augmented_df = pd.read_csv(augmented_path)

model.eval()

relabelled_df = augmented_df.copy()
for idx, sample in tqdm(augmented_df.iterrows(), total=len(augmented_df), desc="Relabeling"):
    inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
        relabelled_df.loc[idx, 'target'] = pred[0]

relabelled_path = os.path.join(DATA_DIR, 'relabelled_data.csv')
relabelled_df.to_csv(relabelled_path, index=False)
print(f"Relabeled data saved to {relabelled_path}")