
import os
import datetime
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import MultiTaskModel
from evaluate import evaluate_multitask_case_level


def train_multitask(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    dropout_rate,
    weight_decay,
    time_loss_weight=2.0,
    log_dir="output_tmp",
    top_k=3,
):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_log_{timestamp}.txt")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    lesion_criterion = nn.CrossEntropyLoss()
    time_criterion = nn.CrossEntropyLoss()

    device = next(model.parameters()).device
    best_val_auc = 0.0
    best_val_metrics = None
    best_model_path = os.path.join(log_dir, "model_best.pt")

    for epoch in range(epochs):
        model.train()
        total_lesion_loss = 0.0
        total_time_loss = 0.0
        lesion_correct = 0
        time_correct_detected = 0
        num_detected_train = 0
        total_samples = 0

        for batch in train_loader:
            if batch is None:
                continue
            images, lesion_labels, time_labels, _ = batch
            images = images.to(device)
            lesion_labels = lesion_labels.to(device)
            time_labels = time_labels.to(device)

            optimizer.zero_grad()
            lesion_out, time_out = model(images)

            lesion_loss = lesion_criterion(lesion_out, lesion_labels)

            pos_mask = lesion_labels == 1
            if pos_mask.any():
                time_loss = time_criterion(time_out[pos_mask], time_labels[pos_mask])
                total_time_loss += time_loss.item()
            else:
                time_loss = torch.tensor(0.0, device=device)

            total_loss = lesion_loss + time_loss_weight * time_loss
            total_loss.backward()
            optimizer.step()

            total_lesion_loss += lesion_loss.item()

            _, lesion_preds = torch.max(lesion_out, 1)
            _, time_preds = torch.max(time_out, 1)

            lesion_correct += (lesion_preds == lesion_labels).sum().item()
            total_samples += images.size(0)

            detected_mask = lesion_preds == 1
            if detected_mask.any():
                num_detected_train += detected_mask.sum().item()
                time_correct_detected += (
                    (time_preds[detected_mask] == time_labels[detected_mask]).sum().item()
                )

        avg_lesion_loss = total_lesion_loss / len(train_loader)
        avg_time_loss = (
            total_time_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        )
        lesion_acc = lesion_correct / total_samples if total_samples > 0 else 0.0
        time_acc_detected = (
            time_correct_detected / num_detected_train if num_detected_train > 0 else 0.0
        )

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            val_metrics = evaluate_multitask_case_level(
                model, val_loader, device, top_k=top_k
            )
            log_line = (
                f"Epoch {epoch+1}:\n"
                f"Train - Lesion Loss: {avg_lesion_loss:.4f}, Time Loss: {avg_time_loss:.4f}\n"
                f"Train - Lesion Acc: {lesion_acc:.4f}, Time Acc (on detected): {time_acc_detected:.4f}\n"
                f"Val   - Time Acc (case-level): {val_metrics['time_acc']:.4f}\n"
                f"Val   - Time AUC (case-level): {val_metrics['time_auc']:.4f}\n"
                f"Val   - Time F1 (case-level): {val_metrics['time_f1']:.4f}\n"
                f"Val   - Time Sens/Spec (case-level): {val_metrics['time_sensitivity']:.4f}/{val_metrics['time_specificity']:.4f}\n"
                f"Val   - Time PPV/NPV (case-level): {val_metrics['time_ppv']:.4f}/{val_metrics['time_npv']:.4f}\n"
            )

            if val_metrics["time_auc"] > best_val_auc:
                best_val_auc = val_metrics["time_auc"]
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), best_model_path)
        else:
            log_line = (
                f"Epoch {epoch+1}:\n"
                f"Train - Lesion Loss: {avg_lesion_loss:.4f}, Time Loss: {avg_time_loss:.4f}\n"
                f"Train - Lesion Acc: {lesion_acc:.4f}, Time Acc (on detected): {time_acc_detected:.4f}\n"
            )

        print(log_line.strip())
        with open(log_filename, "a") as f:
            f.write(log_line)

        scheduler.step()

    final_model_path = os.path.join(log_dir, "model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    return log_filename, final_model_path, best_model_path, best_val_metrics
