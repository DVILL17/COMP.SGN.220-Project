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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
    batch_size = 1
    epochs = 30
    learning_rate = 1e-4

    cnn_1_channels = 16
    cnn_1_kernel = 5
    cnn_1_stride = 2
    cnn_1_padding = 2

    pooling_1_kernel = 3
    pooling_1_stride = 1
    # Output size: b_size x 16 x 214 x 18

    cnn_2_channels = 32
    cnn_2_kernel = 5
    cnn_2_stride = 2
    cnn_2_padding = 2

    pooling_2_kernel = 3
    pooling_2_stride = 2
    # Output size: b_size x 32 x 53 x 4

    classifier_input_features = 64
    classifier_output = 4
    
    # Instantiate our DNN
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

    # Load data
    data_path = 'maestro-v3.0.0'
    
    ds_train = MyDataset(
        data_path,
        data_path + '/maestro-v3.0.0.csv',
        'train'
    )
    ds_val = MyDataset(
        data_path,
        data_path + '/maestro-v3.0.0.csv',
        'validation'
    )
    ds_test = MyDataset(
        data_path,
        data_path + '/maestro-v3.0.0.csv',
        'test'
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size)
            
    
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 10
    patience_counter = 0

    val_losses, test_losses, test_accs = [], [], []
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
        for i, batch in enumerate(train_loader):
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x, y = batch

            # Give them to the appropriate device.
            x = x.to(device)
            y = y.to(device)

            # Flatten the target to match the shape of the model's output.
            y = y.view(1, -1)

            # Get the predictions of our model.
            y_hat = model(x).squeeze(1)
            
            # Flatten the model output
            y_hat = y_hat.view(batch_size, -1)

            # Calculate the loss of our model.
            loss = loss_function(y_hat, y)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Collect predictions and ground truths for accuracy calculation
            train_pred.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
            train_gt.extend(torch.argmax(y, dim=1).cpu().numpy())
            
            epoch_loss_training.append(loss.item())

        model.eval()
        with torch.no_grad():
            # Validation loop
            for i, batch in enumerate(val_loader):
                # Get the batch
                x_val, y_val = batch
                # Pass the data to the appropriate device.
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_gt.extend(torch.argmax(y_val, dim=1).cpu().numpy())

                # Get the predictions of the model.
                y_hat = model(x_val).squeeze(1)
                val_pred.extend(torch.argmax(y_hat, dim=1).cpu().numpy())

                # Flatten the target to match the shape of the model's output.
                y_val = y_val.view(1, -1)

                # Calculate the loss.
                loss = loss_function(y_hat, y_val)

                # Log the validation loss.
                epoch_loss_validation.append(loss.item())
                val_losses.append(loss.item())

        # Calculate mean losses and accuracy.
        epoch_loss_training = np.mean(epoch_loss_training)
        epoch_loss_validation = np.mean(epoch_loss_validation)
        epoch_acc_training = 100 * np.sum(np.array(train_pred) == np.array(train_gt)) / len(train_gt)
        epoch_acc_validation = 100 * np.sum(np.array(val_pred) == np.array(val_gt)) / len(val_gt)

        # Print training and validation info
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training loss: {epoch_loss_training:.4f} | Training accuracy: {epoch_acc_training:.2f}%")
        print(f"Validation loss: {epoch_loss_validation:.4f} | Validation accuracy: {epoch_acc_validation:.2f}%")

        # Early stopping conditions
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if patience_counter >= patience or epoch == epochs - 1:
            print('\nExiting training', end='\n\n')
            print(f'Best epoch {best_validation_epoch+1} with loss {lowest_validation_loss:.4f}', end='\n\n')
            if best_model is None:
                print('No best model.')
            else:
                print('Starting testing', end=' | ')
                testing_loss = []
                testing_acc = 0
                pred = []
                gt = []

                # Load best model
                model.load_state_dict(best_model)
                model.eval()
                with torch.no_grad():
                    for i, batch in enumerate(test_loader):
                        x_test, y_test = batch
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)
                        gt_batch = torch.argmax(y_test, dim=1).cpu().numpy()
                        gt.extend(gt_batch)

                        y_hat = model(x_test).squeeze(1)
                        pred_batch = torch.argmax(y_hat, dim=1).cpu().numpy()
                        pred.extend(pred_batch)

                        # Flatten the target to match the shape of the model's output.
                        y_test = y_test.view(1, -1)

                        loss = loss_function(y_hat, y_test)

                        testing_loss.append(loss.item())
                        test_losses.append(loss.item())

                        # get test accuracy for batch and save
                        testing_acc_batch = np.sum(np.array(pred_batch) == np.array(gt_batch)) / len(gt_batch)
                        testing_acc += testing_acc_batch

                testing_acc = 100 * testing_acc / len(test_loader)
                testing_loss = np.mean(testing_loss)
                print(f'Testing loss: {testing_loss:7.4f} | Testing accuracy: {testing_acc:.2f}%')

                # Show confusion matrix
                cm = confusion_matrix(gt, pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ds_test.labels, yticklabels=ds_test.labels)
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title("Confusion Matrix on Test Data")
                plt.show()
                break


    print(f'Validation losses: {val_losses}')
    print(f'Testing losses: {test_losses}')
    print(f'Testing accuracies: {test_accs}')

    print(f'Validation losses mean: {np.mean(val_losses):7.4f}')
    print(f'Testing losses mean: {np.mean(test_losses):7.4f}')
    print(f'Testing accuracies mean: {np.mean(test_accs):7.4f}', end='\n\n')


if __name__ == '__main__':
    main()
