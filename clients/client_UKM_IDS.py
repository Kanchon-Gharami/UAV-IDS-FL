# clients/client_UKM_IDS.py

import flwr as fl
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import json

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoder_classifier_cnn_lstm import EncoderClassifierCNNLSTM, ReTeLU
from preprocessing.preprocess_UKM_IDS import get_UKM_IDS_dataloaders
from utils.metrics import compute_metrics  # Ensure you have this utility implemented
from utils.ewc import EWC  # Ensure you have the EWC class implemented

import random
import numpy as np

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class UKMIDSClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, label_encoder, device=torch.device("cpu")):
        """
        Initializes the Flower client for UKM_IDS.

        Parameters:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.
            label_encoder (LabelEncoder): Label encoder for the dataset.
            device (torch.device): Device to perform computations on.
        """
        set_seed(42)  # Ensure consistent initialization across clients
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.label_encoder = label_encoder
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.current_round = 0  # To keep track of rounds
        self.ewc = None
        self.lambda_ewc = 0.4  # Regularization strength, adjust as needed

    def get_parameters(self, config):
        """
        Returns the shared encoder parameters.

        Parameters:
            config (dict): Configuration dictionary.

        Returns:
            list: List of NumPy arrays representing shared encoder parameters.
        """
        shared_encoder_params = []
        for name, param in self.model.named_parameters():
            if 'shared_encoder' in name or 'cnn' in name or 'lstm' in name or 'fc_shared' in name:
                shared_encoder_params.append(param.detach().cpu().numpy())
        return shared_encoder_params

    def set_parameters(self, parameters):
        """
        Sets the shared encoder parameters.

        Parameters:
            parameters (list): List of NumPy arrays representing shared encoder parameters.

        Raises:
            ValueError: If there is a mismatch in the number of parameters.
        """
        state_dict = self.model.state_dict()
        shared_encoder_keys = [name for name, param in self.model.named_parameters()
                               if 'shared_encoder' in name or 'cnn' in name or 'lstm' in name or 'fc_shared' in name]
        if len(shared_encoder_keys) != len(parameters):
            raise ValueError("Mismatch between number of shared encoder keys and incoming parameters.")
        for key, val in zip(shared_encoder_keys, parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """
        Trains the model on the client's local dataset.

        Parameters:
            parameters (list): List of NumPy arrays representing shared encoder parameters from the server.
            config (dict): Configuration dictionary.

        Returns:
            tuple: (updated_parameters, num_examples, {})
        """
        self.current_round += 1  # Increment the round counter
        self.set_parameters(parameters)
        self.model.train()

        # Training loop
        for epoch in range(1):  # Adjust epochs if needed
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = F.cross_entropy(y_pred, y)

                if self.ewc is not None:
                    loss += self.lambda_ewc * self.ewc.penalty(self.model)

                loss.backward()
                self.optimizer.step()

        # After training, compute EWC for this round
        self.ewc = EWC(self.model, self.train_loader, self.device)

        # Evaluate after each training round
        avg_loss, total_examples, metrics = self.evaluate_model()
        print(f"UKM Round {self.current_round}")
        print(f"Accuracy: {metrics['accuracy']:.6f}")
        print(f"Precision: {metrics['precision']:.6f}")
        print(f"Recall: {metrics['recall']:.6f}")
        print(f"F1 Score: {metrics['f1_score']:.6f}")
        print("--------------------")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Evaluates the model on the client's local test dataset.

        Parameters:
            parameters (list): List of NumPy arrays representing shared encoder parameters from the server.
            config (dict): Configuration dictionary.

        Returns:
            tuple: (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)
        avg_loss, total_examples, metrics = self.evaluate_model()
        return avg_loss, total_examples, metrics

    def evaluate_model(self):
        """
        Evaluates the model and computes metrics.

        Returns:
            tuple: (avg_loss, total_examples, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        y_true, y_pred = [], []
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = criterion(output, y)
                total_loss += loss.item() * y.size(0)
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == y).sum().item()
                total_examples += y.size(0)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        avg_loss = total_loss / total_examples
        accuracy = total_correct / total_examples

        # Calculate metrics
        accuracy_score, precision, recall, f1 = compute_metrics(y_true, y_pred, average='macro')

        # Save confusion matrix and metrics
        self.save_results(y_true, y_pred, accuracy_score, precision, recall, f1)

        metrics = {"accuracy": accuracy_score, "precision": precision, "recall": recall, "f1_score": f1}
        return avg_loss, total_examples, metrics

    def save_results(self, y_true, y_pred, accuracy, precision, recall, f1):
        """
        Saves confusion matrix and evaluation metrics to disk.

        Parameters:
            y_true (list): True labels.
            y_pred (list): Predicted labels.
            accuracy (float): Accuracy score.
            precision (float): Precision score.
            recall (float): Recall score.
            f1 (float): F1 score.
        """
        # Save confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        label_names = self.label_encoder.classes_

        # Print Confusion Matrix to Terminal
        conf_matrix_df = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)
        print("\nConfusion Matrix:")
        print(conf_matrix_df)
        print("--------------------")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f'Confusion Matrix - Round {self.current_round}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        result_dir = f"result/client_UKM_IDS/round_{self.current_round}"
        os.makedirs(result_dir, exist_ok=True)
        plt.savefig(f"{result_dir}/confusion_matrix_round_{self.current_round}.png")
        plt.close()

        # Save evaluation metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        with open(f"{result_dir}/evaluation_metrics_round_{self.current_round}.json", "w") as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    # Prepare data paths
    train_file = "data/ukm_ids/train.csv"
    test_file = "data/ukm_ids/test.csv"

    # Check if data files exist
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Train or test data files not found at {train_file} and {test_file}.")
        sys.exit(1)

    # Load data
    train_loader, test_loader, input_dim, num_classes, label_encoder = get_UKM_IDS_dataloaders(
        train_file=train_file,
        test_file=test_file,
        batch_size=32
    )

    # Initialize model
    hidden_dim = 128  # Must be divisible by sequence_length=1
    latent_dim = 64   # Adjust as needed
    cnn_channels = 16  # Adjust as needed
    cnn_kernel_size = 3  # Adjust as needed
    lstm_hidden_size = 64  # Adjust as needed
    lstm_num_layers = 2    # Adjust as needed
    sequence_length = 1    # Since data isn't sequential

    # Ensure hidden_dim is divisible by sequence_length
    if hidden_dim % sequence_length != 0:
        raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by sequence_length ({sequence_length}).")

    model = EncoderClassifierCNNLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        cnn_kernel_size=cnn_kernel_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        sequence_length=sequence_length
    )

    # Initialize Flower client with the correct class
    client = UKMIDSClient(model, train_loader, test_loader, label_encoder)

    # Start Flower client using the updated start_client method with .to_client()
    fl.client.start_client(
        server_address="localhost:8081",
        client=client.to_client()
    )
