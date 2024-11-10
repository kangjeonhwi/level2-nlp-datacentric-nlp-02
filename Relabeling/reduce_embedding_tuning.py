# 기본 데이터 처리 및 유틸리티 라이브러리
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# PyTorch 관련 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Hugging Face Transformers 관련 라이브러리
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup

# Optuna (하이퍼파라미터 최적화)
import optuna

# 사용자 정의 모듈
from reduce_embedding import CustomDataset, ReduceDimention, ContrastiveLoss, create_weighted_sampler



def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument("--study_name", type=str,
                        help="Name of the Optuna study for hyperparameter optimization.")
    parser.add_argument("--storage_name", type=str,
                        help="Storage path for the Optuna study (e.g., 'sqlite:///example.db').")
    parser.add_argument("--pretrained_path", type=str,
                        help="Path to the pretrained model to be used as a starting point for training.")
    parser.add_argument("--data_path", type=str,
                        help="Path to the CSV file containing training data.")
    parser.add_argument("--n_trials", type=int,
                        help="Number of trials for Optuna hyperparameter optimization.")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs for training. Default is 10.")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training. Default is 32.")
    args = parser.parse_args()

    model_path = args.pretrained_path
    epochs = args.epochs
    batch_size = args.batch_size
    def objective(trial):
        # 하이퍼파라미터 선택
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3) # 0.00001 ~ 0.001
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        margin = trial.suggest_float("margin", 1.0, 3.0)
        
        # 스케줄러 유형 선택
        scheduler_type = trial.suggest_categorical("scheduler_type", ["linear", "cosine", "step"])

        # FC 레이어의 수 및 크기 설정
        num_fc_layers = trial.suggest_int("num_fc_layers", 2, 6)  # FC 레이어 수를 1~4 사이에서 선택
        
        # 순차적으로 크기를 일정 비율로 감소        
        fc_size_factor = [] 
        for i in range(num_fc_layers) :
            factor = trial.suggest_float(f"factor_{i}", 0.5, 0.8) # 감소 비율 
            fc_size_factor.append(factor)        

        # 데이터 및 모델 설정
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        dataset = pd.read_csv(args.data_path)
        dataset = dataset.iloc[:1600,:]
        train_df, val_df = train_test_split(dataset, test_size=0.1, random_state=456)
        
        train_dataset = CustomDataset(train_df, tokenizer)
        val_dataset = CustomDataset(val_df, tokenizer)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_sampler = create_weighted_sampler(train_dataset)
        
        train_loader = DataLoader(train_dataset,
                                batch_size = batch_size,
                                #shuffle = True,
                                collate_fn = data_collator,
                                sampler=train_sampler)
        val_loader = DataLoader(val_dataset,
                                batch_size = batch_size,
                                collate_fn = data_collator)
        
        # 모델 초기화
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = ReduceDimention(model_path, fc_size_factor, dropout_rate).to(DEVICE)
        criterion = ContrastiveLoss(margin)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
        
        # 스케줄러 설정
        total_steps = epochs * len(train_loader)
        if scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.2), num_training_steps=total_steps)
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        elif scheduler_type == "step":
            step_size = trial.suggest_int("step_size", 10, 50)  # StepLR의 step_size도 튜닝할 수 있음
            scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
        
        # 학습 및 검증 루프
        best_val_loss = float("inf")
        for _ in range(epochs):
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                token_type_ids = batch['token_type_ids'].to(DEVICE)
                labels = batch['target'].to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
            
            # 검증 루프
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    token_type_ids = batch['token_type_ids'].to(DEVICE)
                    labels = batch['target'].to(DEVICE)
                    
                    outputs = model(input_ids, token_type_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        return best_val_loss


    # Optuna 스터디 생성 및 실행 (SQLite 데이터베이스에 저장)
    study = optuna.create_study(study_name=args.study_name, # 스터디 이름
                                storage=args.storage_name, # SQLite 파일 경로
                                direction="minimize",
                                load_if_exists=True
                                )
        
    with tqdm(total=args.n_trials) as pbar :
        def wrapped_objective(trial) :
            result = objective(trial)
            pbar.update(1)
            return result

        study.optimize(wrapped_objective, n_trials=args.n_trials)

    # 최적의 하이퍼파라미터 출력
    print("Best Hyperparameters:", study.best_params)


if __name__ == "__main__":
    main()