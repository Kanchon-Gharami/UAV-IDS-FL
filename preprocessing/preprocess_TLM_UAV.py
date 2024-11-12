# preprocessing/preprocess_TLM_UAV.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class TLMDataset(Dataset):
    def __init__(self, data_path):
        """
        Initializes the TLMDataset.

        Parameters:
            data_path (str): Path to the CSV data file.
        """
        # Load data
        data = pd.read_csv(data_path)

        # Handle missing values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        # Identify feature columns and label column
        feature_columns = data.columns[:-1]  # All columns except the last
        label_column = data.columns[-1]      # Last column

        # Separate features and labels
        X = data[feature_columns]
        y = data[label_column]

        # Check for non-numeric columns in features
        non_numeric_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
        if non_numeric_cols:
            print(f"Non-numeric columns in TLM features: {non_numeric_cols}")
            # Drop non-numeric columns
            X = X.drop(columns=non_numeric_cols)

        # Now X should contain only numerical values
        self.X = X.values

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.y = self.label_encoder.transform(y)

        # Scale features
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
        self.scaler = scaler  # Save scaler if needed

        # Convert to tensors
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]

def get_TLM_UAV_dataloaders(train_file, test_file, batch_size=32):
    """
    Creates DataLoaders for training and testing TLM_UAV dataset.

    Parameters:
        train_file (str): Path to the training CSV file.
        test_file (str): Path to the testing CSV file.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, test_loader, input_dim, num_classes, label_encoder)
    """
    # Create datasets
    train_dataset = TLMDataset(train_file)
    test_dataset = TLMDataset(test_file)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Input dimension and number of classes
    input_dim = train_dataset.X_tensor.shape[1]
    num_classes = len(train_dataset.label_encoder.classes_)

    label_encoder = train_dataset.label_encoder

    return train_loader, test_loader, input_dim, num_classes, label_encoder
