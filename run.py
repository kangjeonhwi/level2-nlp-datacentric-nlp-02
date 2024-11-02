import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import Dataset

import evaluate
import wandb
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }

    def __len__(self):
        return len(self.labels)

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", type = bool, default=True, 
                        help="False if wandb is not used.")
    parser.add_argument("--project_name", type = str, 
                        help = "Project name in wandb.")
    parser.add_argument("--data_dir", type = str,
                        help = "Directory where all data exists.")
    parser.add_argument("--output_dir", type = str,
                        help = "A path to save the results.")

    args = parser.parse_args()

    SEED = 456
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # wandb 설정
    if args.use_wandb:
        wandb.init(project=args.project_name)
        wandb.config.update(args)  # argparse 인자들을 wandb에 업로드
    else :
        os.environ['WANDB_DISABLED'] = 'true'

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_name = 'klue/bert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    data = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    dataset_train, dataset_valid = train_test_split(data, test_size=0.3, random_state=SEED)
    data_train = BERTDataset(dataset_train, tokenizer)
    data_valid = BERTDataset(dataset_valid, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    f1 = evaluate.load('f1')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1.compute(predictions=predictions, references=labels, average='macro')

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy='steps',
        evaluation_strategy = 'steps',
        #eval_strategy='steps',
        save_strategy='steps',
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        learning_rate= 2e-05,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    dataset_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    model.eval()
    preds = []

    for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)
    
    dataset_test['target'] = preds
    dataset_test.to_csv(os.path.join(args.output_dir, "output.csv"), index=False)

    del model
    del inputs
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()