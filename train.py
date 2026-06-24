import torch.nn as nn
import torch
import argparse
from clickbait_detector import create_dataloader, train, ClickBaitDataset, Model

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default=r"./data/raw/train_clickbait.csv")
    parser.add_argument("--val_path", type=str, default=r"./data/raw/val_clickbait.csv")
    parser.add_argument("--test_path", type=str, default=r"./data/raw/test_clickbait.csv")
    parser.add_argument("--save_path", type=str, default=r"./artifacts")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    train_dataset = ClickBaitDataset(train_path)
    val_dataset = ClickBaitDataset(val_path)
    test_dataset = ClickBaitDataset(test_path)
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(val_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")

    train_loader, val_loader, test_loader = create_dataloader(train_dataset, val_dataset, test_dataset)

    model = Model()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train(model, train_loader, val_loader, loss_fn, optimizer, args.epochs, args.save_path)
