# models/encoder_classifier_cnn_lstm.py

import torch
import torch.nn as nn

class ReTeLU(nn.Module):
    """
    Custom Activation Function: Combination of ReLU and Tanh.
    Applies ReLU followed by Tanh to capture both sparsity and bounded non-linearity.
    """
    def __init__(self):
        super(ReTeLU, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        return self.tanh(self.relu(x))

class EncoderClassifierCNNLSTM(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        latent_dim, 
        num_classes, 
        cnn_channels=16, 
        cnn_kernel_size=3, 
        lstm_hidden_size=64, 
        lstm_num_layers=2,
        sequence_length=1  # Must divide hidden_dim
    ):
        super(EncoderClassifierCNNLSTM, self).__init__()
        
        self.sequence_length = sequence_length  # Store for validation
        
        # Client-specific Input Layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ReTeLU()  # ReTeLU Activation
        )
        
        # Calculate channels for CNN and LSTM
        self.cnn_channels = hidden_dim // sequence_length  # 128 //1=128
        self.lstm_channels = hidden_dim // sequence_length  # 128 //1=128
        
        # Shared CNN Path
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=self.cnn_channels, 
                out_channels=cnn_channels,      #16
                kernel_size=cnn_kernel_size,    #3
                padding=1
            ),
            nn.BatchNorm1d(cnn_channels),  #16
            ReTeLU(),  # ReTeLU Activation
            nn.MaxPool1d(kernel_size=2, ceil_mode=True)  # ceil_mode=True to prevent sequence length from becoming zero
        )
        self.cnn_out_channels = cnn_channels  #16
        
        # Shared LSTM Path
        self.lstm = nn.LSTM(
            input_size=self.lstm_channels, 
            hidden_size=lstm_hidden_size, 
            num_layers=lstm_num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Shared Fully Connected Layer to Latent Dimension
        # After MaxPool1d with kernel_size=2, ceil_mode=True: new_sequence_length_cnn = ceil(sequence_length / 2)
        # For sequence_length=1: new_sequence_length_cnn = 1
        self.latent_dim = lstm_hidden_size * 2 + self.cnn_out_channels * ((sequence_length + 1) // 2)  #64*2 +16*1=144
        self.fc_shared = nn.Sequential(
            nn.Linear(self.latent_dim, latent_dim),
            ReTeLU()  # ReTeLU Activation
        )
        
        # Local Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            ReTeLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Client-specific Input Layer
        x = self.input_layer(x)  # Shape: (batch_size, hidden_dim)
        # print(f"After input_layer: {x.shape}")  # Debug
        
        # Reshape for CNN: (batch_size, cnn_channels, sequence_length)
        batch_size = x.size(0)
        hidden_dim = x.size(1)
        sequence_length = self.sequence_length
        assert hidden_dim % sequence_length == 0, f"hidden_dim ({hidden_dim}) must be divisible by sequence_length ({sequence_length})."
        channels_cnn = self.cnn_channels
        x_cnn = x.view(batch_size, channels_cnn, sequence_length)  # Shape: (batch_size, channels_cnn, sequence_length)
        # print(f"After reshaping for CNN: {x_cnn.shape}")  # Debug
        
        # Shared CNN Path
        x_cnn = self.cnn(x_cnn)  # Shape: (batch_size, 16, 1)
        new_sequence_length_cnn = x_cnn.size(2)
        assert new_sequence_length_cnn > 0, f"After CNN and pooling, sequence_length became {new_sequence_length_cnn}, which is invalid."
        # Flatten CNN output
        x_cnn_flat = x_cnn.view(batch_size, self.cnn_out_channels * new_sequence_length_cnn)  # Shape: (batch_size, 16)
        # print(f"After flattening CNN output: {x_cnn_flat.shape}")  # Debug
        
        # Shared LSTM Path
        channels_lstm = self.lstm_channels
        x_lstm = x.view(batch_size, sequence_length, channels_lstm)  # Shape: (batch_size, 1, 128)
        # print(f"Before LSTM: {x_lstm.shape}")  # Debug
        lstm_out, _ = self.lstm(x_lstm)  # Shape: (batch_size, 1, 128)
        # print(f"After LSTM: {lstm_out.shape}")  # Debug
        # Take the last output of LSTM
        lstm_last = lstm_out[:, -1, :]  # Shape: (batch_size, 128)
        # print(f"After selecting last LSTM output: {lstm_last.shape}")  # Debug
        
        # Concatenate CNN and LSTM outputs
        combined = torch.cat((lstm_last, x_cnn_flat), dim=1)  # Shape: (batch_size, 144)
        # print(f"After concatenation: {combined.shape}")  # Debug
        
        # Shared Fully Connected Layer
        latent = self.fc_shared(combined)  # Shape: (batch_size, 64)
        # print(f"After fc_shared: {latent.shape}")  # Debug
        
        # Local Classifier
        out = self.classifier(latent)  # Shape: (batch_size, num_classes)
        # print(f"After classifier: {out.shape}")  # Debug
        
        return out

