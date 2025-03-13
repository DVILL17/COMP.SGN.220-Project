import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from cnn_system import MyCNNSystem
from dataset_class import MyDataset
from prettycm import confusion_matrix as pretty_confusion_matrix
from prettycm import palette


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Evaluate the model on the test dataset.

    :param model: Trained model to evaluate.
    :param test_loader: DataLoader for the test dataset.
    :param device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
    :param threshold: Threshold for binary classification.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", ncols=100):
            x_test, y_test = batch
            x_test, y_test = x_test.to(device), y_test.to(device)

            # Forward pass
            y_hat = model(x_test)

            # Apply sigmoid to get probabilities
            y_pred_prob = torch.sigmoid(y_hat)

            # Apply threshold to get binary predictions
            y_pred_binary = (y_pred_prob > threshold).float()

            # Collect predictions, probabilities, and labels
            all_preds.extend(y_pred_binary.cpu().numpy().flatten())
            all_probs.extend(y_pred_prob.cpu().numpy().flatten())
            all_labels.extend(y_test.cpu().numpy().flatten())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    # Confusion matrix using pretty-confusion-matrix
    cm = pretty_confusion_matrix(
        np.array([
            [np.sum(all_labels == 0) - np.sum(all_preds[all_labels == 0] != 0),  # True Negative
             np.sum(all_preds[all_labels == 0] != 0)],  # False Positive
            [np.sum(all_preds[all_labels == 1] == 0),  # False Negative
             np.sum(all_preds[all_labels == 1] == 1)]  # True Positive
        ])
    )
    cm.set_classname(["Negative", "Positive"])
    cm.set_title("Confusion Matrix")

    # Define palette and draw
    pset = palette(size=5, color="blue")
    pset.draw(cm, "confusion_matrix.png")

if __name__ == '__main__':
    # Check if CUDA is available, else use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Load the best model
    model = MyCNNSystem(
        cnn_channels_out_1=16,
        cnn_kernel_1=(5, 5),
        cnn_stride_1=(2, 2),
        cnn_padding_1=(2, 2),
        pooling_kernel_1=(3, 3),
        pooling_stride_1=(1, 1),
        cnn_channels_out_2=32,
        cnn_kernel_2=(5, 5),
        cnn_stride_2=(2, 2),
        cnn_padding_2=(2, 2),
        pooling_kernel_2=(3, 3),
        pooling_stride_2=(2, 2),
        classifier_input_features=229,
        output_classes=1  # Binary classification
    )
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model = model.to(device)

    # Load test dataset
    ds_test = MyDataset('test')
    test_loader = DataLoader(ds_test, batch_size=1, num_workers=4, pin_memory=True)

    # Evaluate the model with different thresholds
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        evaluate_model(model, test_loader, device, threshold=threshold)