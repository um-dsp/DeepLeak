import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, testloader, criterion=None):
    model.eval()
    correct, total = 0, 0
    counter = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            counter += 1

    return counter, correct / total

def evaluate_model(model, test_loader, verbose=False, roc_data_shadow=[], xai=""):
    model.eval()
    true_labels, predictions, probabilities = [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(outputs.cpu().numpy())

    accuracy = correct / total
    conf_matrix = confusion_matrix(true_labels, predictions, labels=[1, 0])
    TP, FP, FN, TN = conf_matrix[0, 0], conf_matrix[1, 0], conf_matrix[0, 1], conf_matrix[1, 1]
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        roc_data_shadow.append({'Explanation Method': xai, 'FPR': fpr, 'TPR': tpr, 'AUC': roc_auc})
        return roc_data_shadow

def evaluate_models_with_majority(models, test_loader, verbose=False, roc_data_shadow=[], xai="", use_majority=True):
    for model in models:
        model.eval()

    true_labels, probabilities = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(inputs.device), labels.to(inputs.device)
            outputs = torch.stack([m(inputs).squeeze() for m in models], dim=1)
            preds = (outputs > 0.5).float()
            confs = torch.abs(outputs - 0.5)

            chosen = []
            for i in range(len(inputs)):
                if use_majority:
                    maj = preds[i].mean().round()
                    idxs = (preds[i] == maj).nonzero(as_tuple=True)[0]
                    weights = confs[i][idxs] if len(idxs) else confs[i]
                    sel = outputs[i][idxs]
                    chosen_val = torch.sum(weights * sel) / torch.sum(weights)
                else:
                    idx = torch.argmax(confs[i])
                    chosen_val = outputs[i, idx]
                chosen.append(chosen_val.cpu())

            probabilities.extend(chosen)
            true_labels.extend(labels.cpu().numpy())

    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)
    final_preds = (probabilities > 0.5).astype(float)

    conf_matrix = confusion_matrix(true_labels, final_preds, labels=[1, 0])
    TP, FP, FN, TN = conf_matrix[0, 0], conf_matrix[1, 0], conf_matrix[0, 1], conf_matrix[1, 1]
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    if verbose:
        print(f"TPR: {TPR * 100:.2f}%, FPR: {FPR * 100:.2f}%")
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        roc_data_shadow.append({'Explanation Method': xai, 'FPR': fpr, 'TPR': tpr, 'AUC': roc_auc})

    return roc_data_shadow
