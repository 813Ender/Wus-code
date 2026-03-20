
import os
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, roc_auc_score


def compute_binary_metrics(y_true, y_prob):

    y_pred = (np.array(y_prob) > 0.5).astype(int)
    acc = np.mean(y_pred == y_true)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    f1 = f1_score(y_true, y_pred)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    sens = tp / (tp + fn + 1e-5)
    spec = tn / (tn + fp + 1e-5)
    ppv = tp / (tp + fp + 1e-5)
    npv = tn / (tn + fn + 1e-5)

    return acc, auc, f1, sens, spec, ppv, npv


def evaluate_multitask_case_level(model, loader, device, top_k=3):

    model.eval()

    case_dict = defaultdict(
        lambda: {
            "lesion_probs": [],
            "lesion_labels": [],
            "time_probs": [],
            "time_labels": [],
            "lesion_preds": [],
        }
    )

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            images, lesion_labels_batch, time_labels_batch, filenames = batch
            images = images.to(device)

            lesion_out, time_out = model(images)

            lesion_probs = torch.softmax(lesion_out, dim=1)[:, 1].cpu().numpy()
            time_probs = torch.softmax(time_out, dim=1)[:, 1].cpu().numpy()

            lesion_labels_batch = lesion_labels_batch.cpu().numpy()
            time_labels_batch = time_labels_batch.cpu().numpy()

            lesion_pred_batch = (lesion_probs > 0.5).astype(int)

            for i in range(len(filenames)):
                relative_path = filenames[i]
                case_id = relative_path.split(os.sep)[0]

                case_dict[case_id]["lesion_probs"].append(lesion_probs[i])
                case_dict[case_id]["lesion_labels"].append(lesion_labels_batch[i])
                case_dict[case_id]["time_probs"].append(time_probs[i])
                case_dict[case_id]["time_labels"].append(time_labels_batch[i])
                case_dict[case_id]["lesion_preds"].append(lesion_pred_batch[i])

    case_time_probs = []
    case_time_labels = []

    for case_id, data in case_dict.items():
        lesion_preds_case = np.array(data["lesion_preds"])
        lesion_probs_case = np.array(data["lesion_probs"])
        time_probs_case = np.array(data["time_probs"])

        detected_idx = np.where(lesion_preds_case == 1)[0]
        if len(detected_idx) > 0:
            k_use = min(top_k, len(detected_idx))
            detected_lesion_probs = lesion_probs_case[detected_idx]
            topk_in_detected = np.argsort(detected_lesion_probs)[::-1][:k_use]
            topk_idx = detected_idx[topk_in_detected]
            case_time_probs.append(np.mean(time_probs_case[topk_idx]))
            case_time_labels.append(data["time_labels"][0])

    if len(case_time_labels) > 1:
        time_metrics = compute_binary_metrics(
            np.array(case_time_labels), np.array(case_time_probs)
        )
    else:
        time_metrics = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return {
        "time_acc": time_metrics[0],
        "time_auc": time_metrics[1],
        "time_f1": time_metrics[2],
        "time_sensitivity": time_metrics[3],
        "time_specificity": time_metrics[4],
        "time_ppv": time_metrics[5],
        "time_npv": time_metrics[6],
    }
