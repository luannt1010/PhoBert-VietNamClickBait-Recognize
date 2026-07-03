import os
import random
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from clickbait_detector import create_data_split, create_dataloader, train, ClickBaitDataset, Model

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA random
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Make CUDA/cuDNN more deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default=r"./data/processed/combined_dataset.csv")
    parser.add_argument("--save_path", type=str, default=r"./artifacts")

    parser.add_argument("--backbone_lr", type=int, default=5e-6)
    parser.add_argument("--classify_lr", type=int, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    root_dir = args.root_dir
    max_len = args.max_len
    batch_size = args.batch_size
    patience = args.patience
    dropout_rate = args.dropout
    backbone_lr = args.backbone_lr
    classify_lr = args.classify_lr

    df = pd.read_csv(root_dir)
    train_df, val_df, test_df = create_data_split(df)
    train_dataset = ClickBaitDataset(train_df, max_len)
    val_dataset = ClickBaitDataset(val_df, max_len)
    test_dataset = ClickBaitDataset(test_df, max_len)
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(val_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")

    train_loader, val_loader, test_loader = create_dataloader(train_dataset, val_dataset, test_dataset, batch_size)

    model = Model(dropout_rate=dropout_rate)

    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": backbone_lr},
        {"params": model.classify1.parameters(), "lr": classify_lr},
        {"params": model.classify2.parameters(), "lr": classify_lr},
        {"params": model.classify3.parameters(), "lr": classify_lr},
        {"params": model.classify4.parameters(), "lr": classify_lr}], weight_decay=0.01)
    train_label_counts = train_df["label"].value_counts()
    pos_weight = torch.tensor([train_label_counts[0] / train_label_counts[1]], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    train(model, train_loader, val_loader, loss_fn, optimizer, args.epochs, args.save_path, scheduler, None if patience==0 else patience)

if __name__ == "__main__":
    main()
