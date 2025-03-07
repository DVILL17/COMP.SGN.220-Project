# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cnn_system import MyCNNSystem
from dataset_class import MyDataset
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm  

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters
    batch_size = 1
    epochs = 30
    learning_rate = 1e-4

    # CNN and classifier parameters
    cnn_1_channels = 16
    cnn_1_kernel = 5
    cnn_1_stride = 2
    cnn_1_padding = 2

    pooling_1_kernel = 3
    pooling_1_stride = 1

    cnn_2_channels = 32
    cnn_2_kernel = 5
    cnn_2_stride = 2
    cnn_2_padding = 2

    pooling_2_kernel = 3
    pooling_2_stride = 2

    classifier_input_features = 229
    classifier_output = 4  # Number of output classes

    # Instantiate model
    model = MyCNNSystem(
        cnn_1_channels,
        (cnn_1_kernel, cnn_1_kernel),
        (cnn_1_stride, cnn_1_stride),
        (cnn_1_padding, cnn_1_padding),
        (pooling_1_kernel, pooling_1_kernel),
        (pooling_1_stride, pooling_1_stride),
        cnn_2_channels,
        (cnn_2_kernel, cnn_2_kernel),
        (cnn_2_stride, cnn_2_stride),
        (cnn_2_padding, cnn_2_padding),
        (pooling_2_kernel, pooling_2_kernel),
        (pooling_2_stride, pooling_2_stride),
        classifier_input_features,
        classifier_output
    )
    model = model.to(device)

    # Load dataset
    ds_train = MyDataset('train')
    ds_val = MyDataset('validation')
    ds_test = MyDataset('test')

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Define loss and optimizer
    loss_function = nn.BCEWithLogitsLoss()  # No need for sigmoid in model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    lowest_validation_loss = float('inf')
    best_validation_epoch = 0
    patience = 10
    patience_counter = 0

    best_model = None
    for epoch in range(epochs):
        epoch_loss_training = []
        epoch_loss_validation = []
        epoch_acc_training = 0
        epoch_acc_validation = 0
        model.train()

        train_pred = []
        train_gt = []
        val_pred = []
        val_gt = []

        # Training loop
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", ncols=100):
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Flatten target to match output shape
            y = y.view(batch_size, -1)

            # Forward pass
            y_hat = model(x).squeeze(1)

            # Compute loss
            loss = loss_function(y_hat, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Apply sigmoid and threshold to get binary predictions
            y_pred_prob = torch.sigmoid(y_hat)
            y_pred_binary = (y_pred_prob > 0.5).float()

            # Collect predictions and labels
            train_pred.extend(y_pred_binary.cpu().numpy().flatten())
            train_gt.extend(y.cpu().numpy().flatten())

            epoch_loss_training.append(loss.item())

        model.eval()
        with torch.no_grad():
            # Validation loop
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{epochs}", ncols=100):
                x_val, y_val = batch
                x_val, y_val = x_val.to(device), y_val.to(device)

                # Flatten target
                y_val = y_val.view(batch_size, -1)

                # Get model predictions
                y_hat = model(x_val).squeeze(1)

                # Compute loss
                loss = loss_function(y_hat, y_val)
                epoch_loss_validation.append(loss.item())

                # Apply sigmoid and threshold
                y_pred_prob = torch.sigmoid(y_hat)
                y_pred_binary = (y_pred_prob > 0.5).float()

                # Collect predictions and labels
                val_pred.extend(y_pred_binary.cpu().numpy().flatten())
                val_gt.extend(y_val.cpu().numpy().flatten())

        # Compute average loss and accuracy
        epoch_loss_training = np.mean(epoch_loss_training)
        epoch_loss_validation = np.mean(epoch_loss_validation)
        epoch_acc_training = 100 * np.mean(np.array(train_pred) == np.array(train_gt))
        epoch_acc_validation = 100 * np.mean(np.array(val_pred) == np.array(val_gt))

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training loss: {epoch_loss_training:.4f} | Training accuracy: {epoch_acc_training:.2f}%")
        print(f"Validation loss: {epoch_loss_validation:.4f} | Validation accuracy: {epoch_acc_validation:.2f}%")

        # Early stopping logic
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0  # Reset the patience counter since validation loss improved
            best_model = deepcopy(model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1  # Increment patience counter because the validation loss didn't improve

        # If the patience limit is exceeded, stop training
        if patience_counter >= patience:
            print(f"\nStopping early at epoch {epoch+1} with best loss {lowest_validation_loss:.4f}\n")
            if best_model:
                print("Loading best model for testing...")
                torch.save(best_model, 'best_model.pt')
                model.load_state_dict(best_model)

                # Testing phase
                model.eval()
                test_losses = []
                test_acc = []
                gt, pred = [], []

                with torch.no_grad():
                    for batch in tqdm(test_loader, desc="Testing", ncols=100):
                        x_test, y_test = batch
                        x_test, y_test = x_test.to(device), y_test.to(device)

                        # Flatten target to match the output shape
                        y_test = y_test.view(batch_size, -1)

                        # Get predictions
                        y_hat = model(x_test).squeeze(1)

                        # Compute loss
                        loss = loss_function(y_hat, y_test)
                        test_losses.append(loss.item())

                        # Apply sigmoid and threshold for binary classification
                        y_pred_prob = torch.sigmoid(y_hat)
                        y_pred_binary = (y_pred_prob > 0.5).float()

                        # Collect predictions and ground truths
                        pred.extend(y_pred_binary.cpu().numpy().flatten())
                        gt.extend(y_test.cpu().numpy().flatten())

                # Compute testing accuracy and loss
                testing_acc = 100 * np.mean(np.array(pred) == np.array(gt))
                testing_loss = np.mean(test_losses)
                print(f"Testing loss: {testing_loss:.4f} | Testing accuracy: {testing_acc:.2f}%")

                # Confusion matrix
                cm = confusion_matrix(gt, pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")
                plt.show()

            # Exit the loop after early stopping
            break

if __name__ == '__main__':
    main()
    