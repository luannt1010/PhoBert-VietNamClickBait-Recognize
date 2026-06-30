import os
import json
import time
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
from clickbait_detector.net import Model

def load_model(weight_path):
    state_dict = torch.load(weight_path)
    model = Model()
    model.load_state_dict(state_dict["model"])
    return model

def create_dataloader(train_dataset, val_dataset, test_dataset, batch_size=8):
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_one_epoch(model, train_pbar, optimizer, loss_fn, device):
    model.train()
    train_running_loss = 0.0
    num_corrects = 0.0
    num_preds = 0.0
    all_preds = []
    all_gts = []
    thresholds = [0.2, 0.4, 0.6, 0.8]
    for target, input_ids, attention_mask in train_pbar:
        target = target.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        loss = loss_fn(outputs, target.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        logits = torch.sigmoid(outputs).squeeze(1)
        preds_50 = (logits >= 0.5).to(torch.int64)
        for threshold in thresholds:
            preds = (logits >= threshold).to(torch.int64)
            all_preds.append(preds.tolist())
            all_gts.append(target.tolist())

        num_corrects += (preds_50 == target).sum().item()
        num_preds += target.size(0)

        train_running_loss += loss.item()


    train_epoch_loss = train_running_loss / len(train_pbar)
    train_epoch_acc = num_corrects / num_preds
    p, r = compute_p_r_threshold(all_preds, all_gts)
    return train_epoch_loss, train_epoch_acc, p, r

def compute_p_r_threshold(all_preds, all_gts):
    p, r = 0, 0
    num_preds = len(all_preds)
    for i in range(num_preds):
        p += precision_score(all_gts[i], all_preds[i], average="macro", zero_division=0)
        r += recall_score(all_gts[i], all_preds[i], average="macro", zero_division=0)
    return p / (num_preds + 1e-6), r / (num_preds + 1e-6)

def evaluate(model, val_pbar, loss_fn, device):
    model.eval()
    val_running_loss = 0.0
    num_corrects = 0.0
    num_preds = 0.0
    all_preds, all_gts = [], []
    thresholds = [0.2, 0.4, 0.6, 0.8]
    with torch.no_grad():
        for target, input_ids, attention_mask in val_pbar:
            target = target.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, target.unsqueeze(1).float())

            logits = torch.sigmoid(outputs).squeeze(1)
            preds_50 = (logits >= 0.5).to(torch.int64)
            for threshold in thresholds:
                preds = (logits >= threshold).to(torch.int64)
                all_preds.append(preds.tolist())
                all_gts.append(target.tolist())

            num_corrects += (preds_50 == target).sum().item()
            num_preds += target.size(0)
            val_running_loss += loss.item()


        val_epoch_loss = val_running_loss / len(val_pbar)
        val_epoch_acc = num_corrects / num_preds
        p, r = compute_p_r_threshold(all_preds, all_gts)
    return val_epoch_loss, val_epoch_acc, p, r

def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, sp, scheduler=None):

    total_time = 0.0
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "tr_p": [], "tr_r": [], "val_p": [], "val_r": []}

    os.makedirs(sp, exist_ok=True)
    model_save_path = os.path.join(sp, "models")
    report_save_path = os.path.join(sp, "reports")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(report_save_path, exist_ok=True)
    best_save_path = os.path.join(model_save_path, "best.pth")
    last_save_path = os.path.join(model_save_path, "last.pth")
    history_save_path = os.path.join(report_save_path, "history.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        start = time.perf_counter()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
        train_epoch_loss, train_epoch_acc, tr_precision, tr_recall = train_one_epoch(model, train_pbar, optimizer,
                                                                               loss_fn, device)
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validating]", leave=False)
        val_epoch_loss, val_epoch_acc, val_precision, val_recall = evaluate(model, val_pbar, loss_fn, device)

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_acc"].append(val_epoch_acc)
        history["tr_p"].append(tr_precision)
        history["val_p"].append(val_precision)
        history["tr_r"].append(tr_recall)
        history["val_r"].append(val_recall)

        end = time.perf_counter()
        epoch_time = (end - start) / 60
        total_time += epoch_time

        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}m: train_loss={train_epoch_loss:.4f} train_acc={train_epoch_acc:.4f} P={tr_precision:.4f} R={tr_recall:.4f} | "
              f"val_loss={val_epoch_loss:.4f} val_acc={val_epoch_acc:.4f} P={val_precision:.4f} R={val_recall:.4f}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_epoch_acc)
            else:
                scheduler.step()

        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "loss": loss_fn.state_dict(), "epoch": epoch}
        if best_val_acc < val_epoch_acc:
            best_val_acc = val_epoch_acc
            torch.save(checkpoint, best_save_path)
        torch.save(checkpoint, last_save_path)

    history["total_time"] = total_time
    with open(history_save_path, "w") as f:
        json.dump(history, f)

# if __name__ == "__main__":
#     all_preds = []
#     a = torch.tensor([1,2,3,4,5], dtype=torch.float32)
#     all_preds.extend(a.cpu().numpy())
#     print(all_preds)