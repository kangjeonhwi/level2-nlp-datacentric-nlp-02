# 기본 데이터 처리 라이브러리
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# PyTorch 관련 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Hugging Face Transformers 관련 라이브러리
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup

# 학습 관리 및 모니터링 도구
import wandb

# 기타
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['cleaned_again_text']  
        target = row['target']

        # 텍스트를 토큰화 (max_length를 설정하지 않음)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding=False,  # 패딩을 여기서 하지 않습니다.
            truncation=False,  # 잘리지 않도록 설정합니다.
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding.get('token_type_ids', torch.tensor([0])).squeeze(),
            'target': torch.tensor(target, dtype=torch.float)
        }


class ReduceDimention(nn.Module) :
    def __init__(self, pre_trained_path, fc_size_factor, dropout_rate) :
        super(ReduceDimention, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained(pre_trained_path)

        # for param in self.pretrained_model.parameters() :
        #     param.requires_grad = False

        self.pretrained_dim = self.pretrained_model.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        layers = []
        last_hidden_size = self.pretrained_dim
        for i, factor in enumerate(fc_size_factor) :
            layers.append(nn.Linear(last_hidden_size, int(last_hidden_size * factor)))
            last_hidden_size = int(last_hidden_size * factor)
            if i < len(fc_size_factor) - 1 :
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(last_hidden_size))
                layers.append(self.dropout)
        self.fc = nn.Sequential(*layers)
            
    def forward(self, input_ids, token_type_ids, attention_mask) :
        model_output = self.pretrained_model(input_ids, token_type_ids, attention_mask)
        text_embed = self.mean_pooling(model_output, attention_mask)
        # output = self.fc(text_embed)
        return text_embed # output

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # 모든 토큰의 임베딩
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class ContrastiveLoss(nn.Module) :
    def __init__(self, margin = 1.0) :
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels) :
        embeddings = F.normalize(embeddings, p=2, dim=1) # 임베딩 정규화 

        #distance_matrix = torch.cdist(embeddings, embeddings, p = 2) # (batch_size x batch_size)
        
        # 코사인 유사도 기반의 contrastive loss
        distance_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

        labels_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() # (batch_size x batch_size)
        # Consists of 0 and 1

        # 같은 라벨(postive) 거리와 다른 라벨(negative) 거리 추출 
        positive_distances = distance_matrix[labels_matrix == 1] # 1-dim
        negative_distances = distance_matrix[labels_matrix == 0] # 1-dim
        
        # Contrastive Loss 계산
        positive_loss = torch.pow(1 - positive_distances, 2).mean() # 같은 라벨 거리 최소화
        negative_loss = torch.pow(F.relu(negative_distances - self.margin), 2).mean()
        loss = positive_loss + negative_loss

        return loss


class MultiClassContrastiveLoss(nn.Module):
    def __init__(self, num_classes, margin=1.0):
        super(MultiClassContrastiveLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, embeddings, labels):
        # Normalize embeddings for stable distance measurement
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute class centers
        centers = []
        for i in range(self.num_classes):
            class_embeddings = embeddings[labels == i]
            if len(class_embeddings) > 0:
                centers.append(class_embeddings.mean(dim=0))
            else:
                centers.append(torch.zeros_like(embeddings[0]))  # 빈 클래스 처리
        centers = torch.stack(centers)

        # Calculate positive (same class) and negative (different class) distances
        loss = 0.0
        for i in range(len(embeddings)):
            current_embedding = embeddings[i]
            current_label = labels[i].long()

            # Positive distance: distance to the center of its own class
            pos_center = centers[current_label]
            pos_distance = torch.pow(F.cosine_similarity(current_embedding, pos_center, dim=0) - 1, 2)

            # Negative distances: distance to centers of other classes
            neg_distances = torch.stack([torch.pow(F.relu(self.margin - F.cosine_similarity(current_embedding, centers[j], dim=0)), 2) 
                                         for j in range(self.num_classes) if j != current_label]).mean()
            
            # Add both positive and negative losses
            loss += pos_distance + neg_distances

        return loss / len(embeddings)


import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraInterClassContrastiveLoss(nn.Module):
    def __init__(self, num_classes, margin=1.0):
        super(IntraInterClassContrastiveLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, embeddings, labels):
        # Normalize embeddings for consistent distance calculation
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Intra-Class Loss: 같은 클래스 내의 거리를 최소화
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        labels_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        positive_similarities = similarity_matrix[labels_matrix == 1]  # 같은 클래스 간 유사도
        negative_similarities = similarity_matrix[labels_matrix == 0]  # 다른 클래스 간 유사도

        # 같은 클래스의 거리를 최소화하는 손실
        intra_class_loss = torch.pow(1 - positive_similarities, 2).mean()

        # 클래스 중심 계산
        centers = []
        for i in range(self.num_classes):
            class_embeddings = embeddings[labels == i]
            if len(class_embeddings) > 0:
                centers.append(class_embeddings.mean(dim=0))
            else:
                centers.append(torch.zeros_like(embeddings[0]))
        centers = torch.stack(centers)

        # Inter-Class Loss: 클래스 중심 간 거리를 margin 이상으로 유지
        inter_class_loss = 0.0
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                dist = F.cosine_similarity(centers[i].unsqueeze(0), centers[j].unsqueeze(0)).squeeze()
                inter_class_loss += torch.pow(F.relu(self.margin - dist), 2)
        inter_class_loss /= (self.num_classes * (self.num_classes - 1)) / 2  # 평균화

        # 최종 손실: Intra-Class와 Inter-Class 손실을 합산
        loss = intra_class_loss + inter_class_loss
        return loss


def create_weighted_sampler(dataframe):
    dataframe = dataframe.data
    class_counts = dataframe['target'].value_counts().to_dict()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = dataframe['target'].map(class_weights).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def save_model(model, save_path) :
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def main() :
    parser = argparse.ArgumentParser()

    # have no default value
    parser.add_argument("--project_name", type=str, 
                        help="Project name in wandb")
    parser.add_argument("--project_sub_name", type=str, 
                        help="The sub name used in wandb")
    parser.add_argument("--pretrained_path", type=str, 
                        help="Path to the pretrained model")
    parser.add_argument("--model_save_path", type=str, 
                        help="Path to save the best model")
    parser.add_argument("--data_path", type=str, 
                        help="Path to the CSV data file")
    
    # have default value    
    parser.add_argument("--epochs", type=int, default=1000, 
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0009, 
                        help="Learning rate for optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        help="For selecting scheduler")
    parser.add_argument("--step_size", type=int, default=10,
                        help="Step size for StepLR")
    parser.add_argument("--dropout_rate", type=float, default=0.25, 
                        help="Dropout rate for the model")
    parser.add_argument("--fc_size_factor", type=list, default=[0.778, 0.575, 0.664], 
                        help="List of sizes for FC layers")
    parser.add_argument("--margin", type=float, default=0.3, 
                        help="Margin for contrastive loss")
    
    args = parser.parse_args()

    SEED = 456
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    wandb.init(project=args.project_name, name = args.project_sub_name)
    wandb.config.update(args)  # argparse 인자들을 wandb에 업로드

    # Dataset 
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    dataset = pd.read_csv(args.data_path)
    dataset = dataset.iloc[:1600,:]
    
    train_dataset, val_dataset = train_test_split(dataset, 
                                                  test_size=0.1, 
                                                  random_state = SEED) 
    train_dataset = CustomDataset(train_dataset, tokenizer)
    val_dataset = CustomDataset(val_dataset, tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_sampler = create_weighted_sampler(train_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size = args.batch_size,
                              #shuffle = True,
                              collate_fn = data_collator,
                              sampler=train_sampler
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size = args.batch_size,
                            collate_fn = data_collator)
    
    # For training
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = ReduceDimention(pre_trained_path=args.pretrained_path,
                            fc_size_factor=args.fc_size_factor,
                            dropout_rate=args.dropout_rate).to(DEVICE)
    
    criterion = ContrastiveLoss(args.margin)
    # criterion = MultiClassContrastiveLoss(7, args.margin)
    optimizer = optim.Adam(model.fc.parameters(), lr = args.learning_rate)
    
    total_steps = args.epochs * len(train_loader)
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(total_steps * 0.2), 
                                                    num_training_steps=total_steps)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    best_val_loss = float("inf")
    patience_counter = 0
    progress_bar = tqdm(total = total_steps, bar_format='{l_bar} | Remaining: {remaining}', ncols=80)
    for epoch in range(args.epochs) :
        total_train_loss = 0
        for step, batch in enumerate(train_loader) :
            model.train()

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            labels = batch['target'].to(DEVICE)

            outputs = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            progress_bar.update(1)

            wandb.log({"Train loss (Step)" : loss.item()})
        
        # One epoch finish
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad() :
            for batch in val_loader :
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                token_type_ids = batch['token_type_ids'].to(DEVICE)
                labels = batch['target'].to(DEVICE)

                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {avg_val_loss}")
        wandb.log({"Epoch": epoch + 1, "Average Train Loss": avg_train_loss, "Validation Loss": avg_val_loss})

        if avg_val_loss < best_val_loss :
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_model(model, args.model_save_path)
        else :
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter}/{args.patience}")
        
        if patience_counter >= args.patience :
            print("Early stopping triggered.")
            break

    
    wandb.finish()


if __name__ == "__main__":
    main()