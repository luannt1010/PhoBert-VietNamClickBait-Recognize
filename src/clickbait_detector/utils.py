import os
import json
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from clickbait_detector.net import Model
from sklearn.metrics import precision_score, recall_score, roc_curve, auc

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

def compute_f1_score(precision, recall):
    return (2*precision*recall)/(precision+recall+1e-6)

def train_one_epoch(model, train_pbar, optimizer, loss_fn, device):
    model.train()
    train_running_loss = 0.0
    num_corrects = 0.0
    num_preds = 0.0
    all_preds, all_gts = [], []
    for target, input_ids, attention_mask in train_pbar:
        target = target.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        loss = loss_fn(outputs, target.unsqueeze(1).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        logits = torch.sigmoid(outputs).squeeze(1)
        preds = (logits >= 0.5).to(torch.int64)
        all_preds.extend(preds.tolist())
        all_gts.extend(target.tolist())
        num_corrects += (preds == target).sum().item()
        num_preds += target.size(0)

        train_running_loss += loss.item()


    train_epoch_loss = train_running_loss / len(train_pbar)
    train_epoch_acc = num_corrects / num_preds
    p = precision_score(all_gts, all_preds, average="macro", zero_division=0)
    r = recall_score(all_gts, all_preds, average="macro", zero_division=0)
    return train_epoch_loss, train_epoch_acc, p, r


def evaluate_one_epoch(model, val_pbar, loss_fn, device):
    model.eval()
    val_running_loss = 0.0
    num_corrects = 0.0
    num_preds = 0.0
    all_preds, all_gts = [], []
    all_probs = []
    with torch.no_grad():
        for target, input_ids, attention_mask in val_pbar:
            target = target.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, target.unsqueeze(1).float())

            logits = torch.sigmoid(outputs).squeeze(1)
            preds = (logits >= 0.5).to(torch.int64)
            all_preds.extend(preds.detach().cpu().tolist())
            all_gts.extend(target.detach().cpu().tolist())
            all_probs.extend(logits.detach().cpu().tolist())
            num_corrects += (preds == target).sum().item()
            num_preds += target.size(0)
            val_running_loss += loss.item()

        val_epoch_loss = val_running_loss / len(val_pbar)
        val_epoch_acc = num_corrects / num_preds
        p = precision_score(all_gts, all_preds, average="macro", zero_division=0)
        r = recall_score(all_gts, all_preds, average="macro", zero_division=0)
    return all_probs, all_gts, val_epoch_loss, val_epoch_acc, p, r

def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, sp, scheduler=None, patience=None):
    best_val_acc = 0.0
    best_epoch = 0
    history = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": [],
               "tr_p": [], "tr_r": [], "val_p": [], "val_r": [], "tr_f1": [], "val_f1": [], "total_time": 0}

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

    val_all_probs, val_all_gts = [], []
    for epoch in range(num_epochs):
        start = time.perf_counter()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
        train_epoch_loss, train_epoch_acc, tr_precision, tr_recall = train_one_epoch(model, train_pbar, optimizer,
                                                                               loss_fn, device)
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validating]", leave=False)
        all_probs, all_gts, val_epoch_loss, val_epoch_acc, val_precision, val_recall = evaluate_one_epoch(model, val_pbar, loss_fn, device)

        train_f1 = compute_f1_score(tr_precision, tr_recall)
        val_f1 = compute_f1_score(val_precision, val_recall)

        val_all_probs.append(all_probs)
        val_all_gts.append(all_gts)

        history["tr_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["tr_acc"].append(train_epoch_acc)
        history["val_acc"].append(val_epoch_acc)
        history["tr_p"].append(tr_precision)
        history["val_p"].append(val_precision)
        history["tr_r"].append(tr_recall)
        history["val_r"].append(val_recall)
        history["tr_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        end = time.perf_counter()
        epoch_time = (end - start) / 60
        history["total_time"] += epoch_time

        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}m: train_loss={train_epoch_loss:.4f} train_acc={train_epoch_acc:.4f} P={tr_precision:.4f} R={tr_recall:.4f} F1={train_f1:.4f} | "
              f"val_loss={val_epoch_loss:.4f} val_acc={val_epoch_acc:.4f} P={val_precision:.4f} R={val_recall:.4f} F1={val_f1:.4f}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_epoch_acc)
            else:
                scheduler.step()

        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "loss": loss_fn.state_dict(), "epoch": epoch}
        if best_val_acc < val_epoch_acc:
            best_val_acc = val_epoch_acc
            best_epoch = epoch
            torch.save(checkpoint, best_save_path)
        torch.save(checkpoint, last_save_path)

        if patience is not None:
            if epoch - best_epoch >= patience:
                print(f"Early stoping at epoch: {epoch+1}.")
                break

    history["val_all_probs"] = val_all_probs
    history["val_all_gts"] = val_all_gts
    show_results(history, report_save_path)
    with open(history_save_path, "w") as f:
        json.dump(history, f)


def annotate_min(ax, x, y, label="Min"):
    idx = min(range(len(y)), key=lambda i: y[i])
    ax.annotate(f"{label}: {y[idx]:.4f}",
                xy=(x[idx], y[idx]), xytext=(0, -25),
                textcoords="offset points", ha="center",
                arrowprops=dict(arrowstyle="->"))


def annotate_max(ax, x, y, label="Max"):
    idx = max(range(len(y)), key=lambda i: y[i])
    ax.annotate(f"{label}: {y[idx]:.4f}",
                xy=(x[idx], y[idx]), xytext=(0, 15),
                textcoords="offset points",ha="center",
                arrowprops=dict(arrowstyle="->"))
def show_results(history, sp=None):
    tr_loss = history["tr_loss"]
    val_loss = history["val_loss"]
    tr_acc = history["tr_acc"]
    val_acc = history["val_acc"]
    tr_p = history["tr_p"]
    val_p = history["val_p"]
    tr_r = history["tr_r"]
    val_r = history["val_r"]
    tr_f1 = history["tr_f1"]
    val_f1 = history["val_f1"]
    num_epochs = [i+1 for i in range(len(tr_loss))]

    # 1. Loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(num_epochs, tr_loss, label="Train Loss")
    ax[0].plot(num_epochs, val_loss, label="Val Loss")
    annotate_min(ax[0], num_epochs, tr_loss, label="Train min")
    annotate_min(ax[0], num_epochs, val_loss, label="Val min")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    # 2. Accuracy
    ax[1].plot(num_epochs, tr_acc, label="Train Accuracy")
    ax[1].plot(num_epochs, val_acc, label="Val Accuracy")
    annotate_max(ax[1], num_epochs, tr_acc, label="Train max")
    annotate_max(ax[1], num_epochs, val_acc, label="Val max")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True)
    if sp is not None:
        fig.savefig(os.path.join(sp, "loss_acc.png"))

    # 3. Precision
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(num_epochs, tr_p, label="Train Precision")
    ax[0].plot(num_epochs, val_p, label="Val Precision")
    annotate_max(ax[0], num_epochs, tr_p, label="Train max")
    annotate_max(ax[0], num_epochs, val_p, label="Val max")
    ax[0].set_title("Precision")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Precision")
    ax[0].legend()
    ax[0].grid(True)

    # 4. Recall
    ax[1].plot(num_epochs, tr_r, label="Train Recall")
    ax[1].plot(num_epochs, val_r, label="Val Recall")
    annotate_max(ax[1], num_epochs, tr_r, label="Train max")
    annotate_max(ax[1], num_epochs, val_r, label="Val max")
    ax[1].set_title("Recall")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Recall")
    ax[1].legend()
    ax[1].grid(True)
    if sp is not None:
        fig.savefig(os.path.join(sp, "precision_recall.png"))

    # 5. F1-score
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(num_epochs, tr_f1, label="Train F1-score")
    ax[0].plot(num_epochs, val_f1, label="Val F1-score")
    annotate_max(ax[0], num_epochs, tr_f1, label="Train max")
    annotate_max(ax[0], num_epochs, val_f1, label="Val max")
    ax[0].set_title("F1-score")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("F1-score")
    ax[0].legend()
    ax[0].grid(True)

    # ROC Curve
    if "val_all_gts" in history and "val_all_probs" in history:
        best_epoch_idx = max(range(len(val_f1)), key=lambda i: val_f1[i])

        y_true = history["val_all_gts"][best_epoch_idx]
        y_prob = history["val_all_probs"][best_epoch_idx]

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        ax[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax[1].plot([1], [1], linestyle="--", label="Random")
        ax[1].set_title(f"ROC Curve - Epoch {best_epoch_idx + 1}")
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[1].legend()
        ax[1].grid(True)

    else:
        ax[1].set_title("ROC Curve")
        ax[1].axis("off")
    if sp is not None:
        fig.savefig(os.path.join(sp, "f1_roc.png"))

    plt.tight_layout()
    plt.show()
