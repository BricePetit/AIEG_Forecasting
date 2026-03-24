"""
In this module, we define the models that will be used in the project.
We have the seq2point (https://dl.acm.org/doi/abs/10.5555/3504035.3504353) model and
the UNet model (https://dl.acm.org/doi/10.1145/3427771.3427859).

This module defines the models using PyTorch.
"""
__title__: str = "pytorch_models"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# Imports standard libraries
from typing import List

# Imports third party libraries
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as f

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- CLASSES ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


class PyTorchSimpleMLP(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int = 64, output_size: int = 1,
        dropout_rate: float = 0.2
    ):
        """
        Simple MLP model with one hidden layer and batch normalization.

        :param input_size:      Size of the input layer.
        :param hidden_size:     Size of the hidden layer.
        :param output_size:     Size of the output layer.
        :param dropout_rate:    Dropout rate.
        """
        super().__init__()
        
        # First layer.
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        # Add only one hidden layer.
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        # Output layer.
        self.fc2 = nn.Linear(hidden_size, output_size)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x:   Input tensor.

        :return:    Output tensor.
        """
        # Verify the shape of the input.
        batch_size = x.shape[0]
        # Flatten sequence dimension (seq_length * num_features).
        x = x = x.reshape(batch_size, -1)
        # Apply the first layer with activation ReLU and dropout.
        x = self.fc1(x)
        x = f.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        # Apply the hidden layer with activation ReLU and dropout.
        x = self.hidden(x)
        x = f.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        # Apply the output layer.
        x = self.fc2(x)
        return x


class PyTorchMLP(nn.Module):
    def __init__(
        self, input_size: int, hidden_layers: List[int] = [128, 64, 32], output_size: int = 1,
        dropout_rate: float = 0.2
    ):
        """
        More complex MLP model with multiple hidden layers and batch normalization.

        :param input_size:      Size of the input layer.
        :param hidden_layers:   List of sizes of the hidden layers.
        :param output_size:     Size of the output layer.
        :param dropout_rate:    Dropout rate.
        """
        super().__init__()
        # Create the list of layers.
        layers = []
        # Input layer.
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.Dropout(dropout_rate))
        # Hidden layers.
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
            layers.append(nn.Dropout(dropout_rate))
        # Output layer.
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.ReLU())
        # Build the sequential model.
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x:   Input tensor.

        :return:    Output tensor.
        """
        # Verify the shape of the input.
        batch_size = x.shape[0]
        # Flatten sequence dimension (seq_length * num_features).
        x = x.view(batch_size, -1) 
        # Apply the model.
        x = self.model(x)
        return x


class PyTorchSeq2Point(nn.Module):
    """
    PyTorch's implementation of the Seq2Point model. We use it as an inner class because we
    won't use it outside the Seq2Point class.
    """

    def __init__(self, in_length: int, out_length: int, features: int,  *args, **kwargs):
        """
        Constructor / Initializer of the Seq2PointPytorchModule class.

        :param in_length:   Length of the input sequence.
        :param out_length:  Length of the output sequence
        :param features:    Number of features in the input tensor.
        """
        self.features = features
        super().__init__(*args, **kwargs)
        # in_channels is the number of channels in the input tensor
        # out_channels is equivalent to the number of filters in tensorflow
        # compute the final size of the sequence after the convolutions
        final_size = in_length - 10 + 1  # After first Conv1d
        final_size = final_size - 8 + 1  # After second Conv1d
        final_size = final_size - 6 + 1  # After third Conv1d
        final_size = final_size - 5 + 1  # After fourth Conv1d
        final_size = final_size - 5 + 1
        self.layers = nn.Sequential(
            nn.Conv1d(self.features, 30, 10, 1),
            nn.ReLU(),
            nn.Conv1d(30, 30, 8, 1),
            nn.ReLU(),
            nn.Conv1d(30, 40, 6, 1),
            nn.ReLU(),
            nn.Conv1d(40, 50, 5, 1),
            nn.ReLU(),
            nn.Conv1d(50, 50, 5, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(50 * final_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_length),
            # nn.ReLU()
        )
        self.apply(init_weights_uniform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function to apply the forward pass.

        :param x:   Tensor containing the input.

        :return:    Return a tensor containing the output.
        """
        return self.layers(x)


class PyTorchUNetNilm(nn.Module):
    """
    PyTorch's implementation of the UNet model. We use it as an inner class because we
    won't use it outside the Seq2Point class.

    Original code: https://github.com/BHafsa/deep-nilmtk-v1
    """

    def __init__(self, features: int, *args, **kwargs):
        """
        Constructor / Initializer of the UNetPytorchModule class.

        :param features:    Number of features in the input tensor.
        """
        super().__init__(*args, **kwargs)
        self.features = features
        pool_filter = 8
        output_size = 1
        # in_channels is the number of channels in the input tensor
        # out_channels is equivalent to the number of filters in tensorflow
        # compute the final size of the sequence after the convolutions
        # 10 -> 8 -> 6 -> 5 -> 5 puis 5 -> 5 -> 6
        # 8 -> 6 -> 4 -> 3 -> 3 puis 3 -> 3 -> 4
        self.enc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.features, 30, 10, 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(30, 30, 8, 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(30, 40, 6, 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(40, 50, 5, 1),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Conv1d(50, 50, 5, 1),
                nn.ReLU()
            )
        ])

        self.dec_layers = nn.ModuleList([
            PyTorchUpLayer(50, 40, 5, 1),
            PyTorchUpLayer(40, 30, 5, 1),
            PyTorchUpLayer(30, 30, 6, 1)
        ])

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(pool_filter),
            nn.Flatten(),
            nn.Linear(30 * pool_filter, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_size)
        )

        self.apply(init_weights_normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function to apply the forward pass.

        :param x:   Tensor containing the input.

        :return:    Return a tensor containing the output.
        """
        xi = [self.enc_layers[0](x)]

        for layer in self.enc_layers[1:]:
            xi.append(layer(xi[-1]))

        for i, layer in enumerate(self.dec_layers):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        out = self.fc(xi[-1])
        return out


class PyTorchUpLayer(nn.Module):
    """
    Class for the Up layer in PyTorch.

    Original code : https://github.com/BHafsa/deep-nilmtk-v1
    """

    def __init__(self, in_chan: int, out_chan: int, kernel: int, stride: int, *args, **kwargs):
        """
        Constructor / Initializer of the Up class.

        :param in_chan:     Input channels.
        :param out_chan:    Output channels.
        :param kernel:      Kernel size.
        :param stride:      Stride.
        """
        super().__init__(*args, **kwargs)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(in_chan, out_chan, kernel, stride),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_chan+out_chan, out_chan, kernel, stride),
            nn.ReLU()
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Function to apply the forward pass.

        :param x1:  Tensor containing the input.
        :param x2:  Tensor containing the input.

        :return:    Return a tensor containing the output.
        """
        x1 = self.upsample(x1)
        # Pad x1 to the size of x2
        diff = x2.shape[2] - x1.shape[2]
        x1 = f.pad(x1, [diff // 2, diff - diff // 2])
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class PyTorchRNN(nn.Module):
    """
    Class for the RNN model in PyTorch.
    """

    def __init__(
            self, input_size: int, out_length: int, hidden_size: int, num_layers: int,
            drop_prob: float, *args, **kwargs
    ):
        """
        Constructor / Initializer of the RNN class.

        :param input_size:  Number of features in the input tensor.
        :param out_length:  Length of the output sequence.
        :param hidden_size: Size of the hidden state.
        :param num_layers:  Number of recurrent layers.
        :param drop_prob:   Dropout probability.
        """
        super().__init__(*args, **kwargs)
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, 'relu',
            dropout=drop_prob if num_layers > 1 else 0.0, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, out_length)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.apply(init_weights_uniform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function to apply the forward pass.

        :param x:   Tensor containing the input.

        :return:    Return a tensor containing the output.
        """
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        # out = self.activation(out)
        return out


class PyTorchLSTM(nn.Module):
    """
    Class for the LSTM model in PyTorch.
    """

    def __init__(
        self, input_size: int, out_length: int, hidden_size: int, num_layers: int,
        drop_prob: float, *args, **kwargs
    ):
        """
        Constructor / Initializer of the LSTM class.

        :param input_size:  Number of features in the input tensor.
        :param out_length:  Length of the output sequence.
        :param hidden_size: Number of features in the hidden state h.
        :param num_layers:  Number of recurrent layers.
        :param drop_prob:   Dropout probability.
        """
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=drop_prob if num_layers > 1 else 0.0,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size, out_length)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.apply(init_weights_uniform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function to apply the forward pass.

        :param x:   Tensor containing the input.

        :return:    Return a tensor containing the output.
        """
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        # out = self.activation(out)
        return out


class PyTorchGRU(nn.Module):
    """
    Class for the GRU model in PyTorch.
    """

    def __init__(
        self, input_size: int, out_length: int, hidden_size: int = 256, num_layers: int = 2,
        drop_prob: float = 0.2, *args, **kwargs
    ):
        """
        Constructor / Initializer of the GRU class.

        :param input_size:  Number of features in the input tensor.
        :param out_length:  Length of the output sequence.
        :param hidden_size: Number of features in the hidden state h.
        :param num_layers:  Number of recurrent layers.
        :param drop_prob:   Dropout probability.
        """
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, dropout=drop_prob if num_layers > 1 else 0.0,
            batch_first=True, bidirectional=True
        )
        # Fully connected layer to map the output of the GRU to the desired output length.
        # We multiply the hidden_size by 2 because we use a bidirectional GRU. 
        self.fc = nn.Linear(hidden_size * 2, out_length)
        self.activation = nn.ReLU()
        self.apply(init_weights_uniform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function to apply the forward pass.

        :param x:   Tensor containing the input.

        :return:    Return a tensor containing the output.
        """
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        # out = self.activation(out)
        return out

class PyTorchCNNGRU(nn.Module):
    def __init__(self, n_features: int = 1, output_size: int = 1):
        """
        Convolutional GRU model for time series data.

        info supplémentaires :
        kernel -> 2-3 pour 8 à 24 points, 3-5 pour 24 à 96 points et 5-7 pour plus de 96
        Padding='same' pour préserver la taille de la séquence
        out_channels = 32-128 pour la plupart des pb et 256-512 pour les plus complexes
        Why batch norm : https://dl.acm.org/doi/10.5555/3045118.3045167
        Pooling output size 1 pour les séquence courte et 5-10 pour les séquences longues
        hidden_size 100-256 pour données complexes, couche 1 pour la plus part des pb et 2-3 pour
        des complexes

        :param n_features:  Number of features in the input tensor.
        :param output_size: Size of the output layer.
        """
        super().__init__()
        # First convolutional layer.
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
        # First batch normalization to the first layer.
        self.bn1 = nn.BatchNorm1d(64)
        # Second convolution layer.
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Second batch normalization.
        self.bn2 = nn.BatchNorm1d(128)
        # Add pooling.
        self.pool = nn.AdaptiveMaxPool1d(output_size=4)
        # Dropout to prevent overfitting.
        self.dropout1 = nn.Dropout(0.2)
        # Bidirectional GRU.
        self.gru = nn.GRU(
            input_size=128, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True
        )
        # Dropout after GRU.
        self.dropout2 = nn.Dropout(0.2)
        # Fully connected layers.
        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=64, out_features=output_size)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PyTorchCNNGRU class.

        :param x:   Input tensor (batch_size, sequence_length, input_size).
        
        :return:    Output tensor (batch_size, output_size).
        """
        # (batch, seq_length, features) -> (batch, features, seq_length).
        x = x.transpose(1, 2).contiguous()
        # Apply convolutions.
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        # Pooling.
        x = self.pool(x)
        x = self.dropout1(x)
        # GRU (attend [batch, seq, channels]).
        x = x.transpose(1, 2).contiguous()
        x, _ = self.gru(x)
        # Get last output from GRU (all directions).
        x = x[:, -1, :]
        x = self.dropout2(x)
        # FC layers.
        x = f.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


class PyTorchComplexCNNGRU(nn.Module):
    def __init__(
        self,
        n_features: int,
        output_size: int = 1,
        cnn_channels: list = [64, 128, 256],
        kernel_size: int = 3,
        gru_hidden_size: int = 256,
        gru_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        pool_size: int = 4,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        # =========================
        # CNN BLOCK
        # =========================
        layers = []
        in_channels = n_features
        for out_channels in cnn_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        # =========================
        # GRU
        # =========================
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        gru_output_size = gru_hidden_size * (2 if bidirectional else 1)
        # =========================
        # FC
        # =========================
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, gru_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size, output_size),
        )

    def forward(self, x):
        # (batch, seq, features) → (batch, features, seq)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.pool(x)
        x = self.dropout(x)
        # (batch, channels, seq) → (batch, seq, channels)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        # last timestep
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class PyTorchTimeSeriesTransformer(nn.Module):
    def __init__(
        self, input_size: int, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int,
        output_size: int, dropout: float = 0.1, causal: bool = True
    ):
        """
        Transformer model for time series data.

        :param input_size:      Size of the input layer.
        :param d_model:         Dimension of the model.
        :param num_heads:       Number of heads in the multi-head attention.
        :param num_layers:      Number of layers in the transformer.
        :param dim_feedforward: Dimension of the feedforward network.
        :param output_size:     Size of the output layer.
        :param dropout:         Dropout rate.
        :param causal:          Whether to use causal masking.
        """
        super().__init__()
        # Linear embedding to transform the input to d_model dimension.
        self.input_embedding = nn.Linear(input_size, d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        # Positional Encoding.
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        # Transformer Encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final linear layer to transform the output to the desired size.
        self.fc_out = nn.Linear(d_model, output_size)
        self.causal = causal

    def generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device) * float("-inf"), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PyTorchTimeSeriesTransformer class.

        :param x:   Input tensor (batch_size, sequence_length, input_size).

        :return:    Output tensor (batch_size, output_size).
        """
        # Linear embedding to transform the input to d_model dimension.
        x = self.input_embedding(x)
        # Batch normalization with permutation to match the expected shape.
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  
        # Positional encoding.
        x = self.positional_encoding(x)
        mask = None
        if self.causal:
            seq_len = x.size(1)
            mask = self.generate_causal_mask(seq_len, x.device)
        # Transformer encoder.
        x = self.transformer_encoder(x, mask=mask)
        # Final linear layer to transform the output to the desired size.
        x = self.fc_out(x[:, -1, :])
        return x


class PositionalEncoding(nn.Module):
    """
    Encode the temporal position to keep the order of the data in the Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Constructor of the PositionalEncoding class.

        :param d_model: Dimension of the model.
        :param max_len: Maximum length of the sequence.
        :param dropout: Dropout rate.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalEncoding class.

        :param x:   Input tensor.

        :return:    Return a tensor containing the output.
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)



def init_weights_uniform(layer):
    """
    Function to initialize the weights with a uniform distribution.

    :param layer:   Layer to initialize.
    """
    if isinstance(layer, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight.data)


def init_weights_normal(layer):
    """
    Function to initialize the weights with a normal distribution.

    :param layer:   Layer to initialize.
    """
    if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_normal_(layer.weight.data)
    elif isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight.data)
