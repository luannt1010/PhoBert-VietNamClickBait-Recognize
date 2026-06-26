import torch
import argparse
import pandas as pd
import torch.nn as nn
from clickbait_detector import create_dataloader, train, ClickBaitDataset, Model

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default=r"./data/raw/train_clickbait.csv")
    parser.add_argument("--val_path", type=str, default=r"./data/raw/val_clickbait.csv")
    parser.add_argument("--test_path", type=str, default=r"./data/raw/test_clickbait.csv")
    parser.add_argument("--save_path", type=str, default=r"./artifacts")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=50)
    return parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    max_len = args.max_len
    batch_size = args.batch_size

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_dataset = ClickBaitDataset(train_df, max_len)
    val_dataset = ClickBaitDataset(val_df, max_len)
    test_dataset = ClickBaitDataset(test_df, max_len)
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(val_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")

    train_loader, val_loader, test_loader = create_dataloader(train_dataset, val_dataset, test_dataset, batch_size)

    model = Model()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_label_counts = train_df["final_label"].value_counts()
    pos_weight = torch.tensor([train_label_counts[0]/train_label_counts[1]], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train(model, train_loader, val_loader, loss_fn, optimizer, args.epochs, args.save_path)
