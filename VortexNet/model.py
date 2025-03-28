"""
Model for Votex Net Graph Attention Network Model. 

Ref: P. Velickovic, https://arxiv.org/abs/1710.10903
Y. Shen, June 2024

Modification:
Y. Shen, August 10 2024: GNN1: model with GAT attention 
Y. Shen, September 20 2024: GNN2: model with edge encoding only
Y. Shen, September 30 2024: GNN3: model with attention and edge encoding
Y. Shen, October 10 2024: GNN4: updated model

"""

import torch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, LayerNorm, \
    BatchNorm, SAGEConv, GINEConv, GATv2Conv, JumpingKnowledge
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn




class GNN4(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels,
                 num_coarse=None, num_fine=None, dropout_rate=0.2, HEADS=6, ALPHA=0.5, HOP=3):
        super().__init__()

        self.HOP = HOP
        self.ALPHA = ALPHA

        # Node and Edge Encoders
        self.node_encoder = nn.Linear(node_in_channels, hidden_channels * hidden_channels)
        self.edge_encoder = nn.Linear(edge_in_channels, hidden_channels)

        # Initialize lists to store layers and output dimensions
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.layer_output_dims = []

        # Create GNN layers with skip connections structure
        for i in range(HOP):
            if i == 0:
                # First layer input dimension
                in_channels = hidden_channels * hidden_channels  # Output of node_encoder
            else:
                in_channels = hidden_channels * HEADS  # Output of previous layer

            if i == HOP - 1:
                # Last layer output dimension
                out_channels_layer = out_channels
                heads = 1
                concat = False
            else:
                out_channels_layer = hidden_channels
                heads = HEADS
                concat = True

            # Define GATv2Conv layer
            conv = GATv2Conv(in_channels, out_channels_layer,
                             edge_dim=hidden_channels, heads=heads, concat=concat)
            self.conv_layers.append(conv)

            # Define BatchNorm layer
            if concat:
                norm_layer = BatchNorm(out_channels_layer * heads)
                layer_dim = out_channels_layer * heads
            else:
                norm_layer = BatchNorm(out_channels_layer)
                layer_dim = out_channels_layer
            self.norm_layers.append(norm_layer)
            self.layer_output_dims.append(layer_dim)

        # Define U-Net-style skip connections and linear layers for dimension adjustment
        self.skip_connections = {}
        self.skip_linear_layers = nn.ModuleDict()

        # Create skip connections
        for i in range(HOP // 2):
            encoder_layer = i
            decoder_layer = HOP - i - 1
            self.skip_connections[encoder_layer] = decoder_layer

            # Get output dimensions
            encoder_dim = self.layer_output_dims[encoder_layer]
            decoder_dim = self.layer_output_dims[decoder_layer]

            if encoder_dim != decoder_dim:
                # Create a linear layer to map from encoder_dim to decoder_dim
                self.skip_linear_layers[str(encoder_layer)] = nn.Linear(encoder_dim, decoder_dim)
            else:
                # No need for linear layer if dimensions match
                self.skip_linear_layers[str(encoder_layer)] = None

        # Optional Fully Connected Layer when dimensions are different
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        if num_coarse != num_fine:
            self.fc = nn.Linear(num_coarse * num_coarse, num_fine * num_fine)
        else:
            self.fc = None

        self.dropout_rate = dropout_rate

    def forward(self, data, return_latent_space = False, 
                modify_latent=False, layer_to_modify=None, 
                modified_latent=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode node and edge features
        x = self.node_encoder(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        edge_attr = self.edge_encoder(edge_attr)

        layer_outputs = []
        encoder_outputs = {}  # To store outputs for skip connections
        latent_space_outputs = []
        # Forward through GNN layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index, edge_attr)
            x = self.norm_layers[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Store x for skip connections if needed (after convolution and activation)
            if i in self.skip_connections:
                encoder_outputs[i] = x


            if return_latent_space:
                latent_space_outputs.append(x.clone())

            if modify_latent and i == layer_to_modify:
                # Replace x with the modified latent representation
                x = modified_latent    
                
            layer_outputs.append(x)    

        # Apply U-Net-style skip connections
        for encoder_layer, decoder_layer in self.skip_connections.items():
            x_encoder = encoder_outputs[encoder_layer]
            x_decoder = layer_outputs[decoder_layer]

            # Get the linear layer for dimension adjustment, if any
            linear_layer = self.skip_linear_layers[str(encoder_layer)]

            if linear_layer is not None:
                x_encoder = linear_layer(x_encoder)

            # Add to decoder layer output
            layer_outputs[decoder_layer] = (1-self.ALPHA)*x_decoder + self.ALPHA*x_encoder

        # Output is from the last layer
        x = layer_outputs[-1]

        # Optional fully connected layer
        if self.fc:
            x = x.view(1, -1)
            x = self.fc(x)
            x = x.view(-1, 1)

        # Return latent space outputs if needed
        if return_latent_space:
            return x, latent_space_outputs
        else:
            return x