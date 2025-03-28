"""
Functions for training the VortexNet model.
Yiren Shen, Sep 2024

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import NNConv
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
import copy
from torch_geometric.data import Batch
import numpy as np

def custom_collate(data_list):
    # collate function for DataLoader for physical AIC-RHS loss
    batch = Batch.from_data_list(data_list)
    batch.aic_matrices = [data.aic_matrix for data in data_list]
    batch.rhs_matrices = [data.rhs_matrix for data in data_list]
    batch.dcpsid_list = [data.dcpsid for data in data_list]
    batch.factor_list = [data.factor for data in data_list]
    batch.chord_list = [data.chord for data in data_list]
    batch.rnmax_list = [data.rnmax for data in data_list]
    return batch


class PenalizedSmoothL1Loss(nn.Module):
    """
    Penalized Smooth L1 Loss function with additional penalties for zero values and sign flips.
    """
    def __init__(self, penalty_weight=1.0):
        super(PenalizedSmoothL1Loss, self).__init__()
        self.huber_loss = nn.SmoothL1Loss()
        self.penalty_weight = penalty_weight
        self.alpha = 10.0

    def forward(self, predictions, targets):
        huber_loss = self.huber_loss(predictions, targets)
        diff = predictions - targets
        squared_error = diff ** 2
        # Emphasize small values
        small_value_emphasis = torch.where(targets < 1e-1, self.alpha * squared_error, squared_error)
        zero_penalty = torch.mean((predictions == 0).float()) * self.penalty_weight
        zero_penalty = 0.0
        sign_flip_penalty = torch.mean((torch.sign(predictions) != torch.sign(targets)).float()) * self.penalty_weight


        return huber_loss + zero_penalty + sign_flip_penalty + small_value_emphasis.mean()

def train_model(model, data, learning_rate=0.01, epochs=1000, clip_value=5000.0, \
                noise_level=0.01, penalty_weight=1.0, descent_batch=10, device='cpu'):
    """
    Train model.
    """
    model.to(device)
    data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-4)
    criterion = PenalizedSmoothL1Loss(penalty_weight=penalty_weight)
    
    # Define the learning rate scheduler
    def lr_lambda(epoch):
        return 0.5 ** (epoch // descent_batch)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=300, factor=0.5)

    def train():
        model.train()
        optimizer.zero_grad()

        # Add random noise to input features
        noisy_x = data.x.clone()
        noisy_x[:, 3] += noise_level * torch.randn_like(data.x[:, 3])
        data.x = noisy_x

        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        scheduler.step(loss)
        return loss.item(), out
    
    # early stopping with patience
    best_loss = float('inf')
    patience = 1000

    for epoch in range(epochs):
        loss, out = train()
        if epoch % 300 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
            print('Predictions:', out[:10].detach().cpu().numpy().flatten())
            print('True values:', data.y[:10].detach().cpu().numpy().flatten())
        
        # early stopping
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
                        
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}')
                break
            
    print('Training complete.')
    return model


def compute_physical_residual(out, data):
    """
    Compute the physical residual: residual = AIC * GAMMA - RHS

    Parameters:
    - out: Model output (predicted DCP values), shape [total_nodes_in_batch, 1]
    - data: Batched Data object containing additional variables

    Returns:
    - total_residual: The sum of the physical residuals for the batch
    """
    device = out.device
    total_residual = 0.0
    num_graphs = data.num_graphs  # Number of samples in the batch
    batch_indices = data.batch    # Tensor indicating graph indices for each node

    for i in range(num_graphs):
        # Get indices of nodes belonging to graph i
        node_indices = (batch_indices == i).nonzero(as_tuple=True)[0]

        # Extract per-graph node outputs
        out_i = out[node_indices].squeeze()

        # Extract per-graph attributes
        dcpsid = data.dcpsid[i].to(device).squeeze()
        chord = data.chord[i].to(device).squeeze()
        rnmax = data.rnmax[i].to(device).squeeze()
        factor = data.factor[i].to(device).squeeze()
        aic_matrix = data.aic_matrix[i].to(device)
        rhs_matrix = data.rhs_matrix[i].to(device).squeeze()

        # Ensure shapes are compatible
        if out_i.dim() == 0:
            out_i = out_i.unsqueeze(0)
        if dcpsid.dim() == 0:
            dcpsid = dcpsid.unsqueeze(0)

        # Compute GNET
        GNET = (out_i - dcpsid) / 2.0

        # Compute GAMMA
        GAMMA = (GNET * (chord / rnmax)) / factor

        # Compute residual
        residual = aic_matrix @ GAMMA.unsqueeze(-1) - rhs_matrix.unsqueeze(-1)

        # Accumulate residual norm
        residual_norm = torch.norm(residual, p=2)
        total_residual += residual_norm

    return total_residual


def train_model_k_fold(model, data, k=4, learning_rate=0.01, epochs=100, batch_size=32, clip_value=5000.0, \
                       noise_level=0.01, penalty_weight=1.0, descent_batch=10, Lambda=1.0, decay=1e-4, \
                        max_phy_loss = 1, device='cpu'):
    """
    Train the model using k-fold cross-validation.

    Args:
        model (nn.Module): The GNN Model.
        data (list): The dataset as a list of data objects.
        k (int): Number of folds for cross-validation.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train for each fold.
        batch_size (int): Batch size for the DataLoader.
        clip_value (float): Gradient clipping value.
        noise_level (float): Noise level added to the input features.
        penalty_weight (float): Weight for the penalty in the loss function.
        descent_batch (int): Frequency of learning rate reduction.
        device (str): The device to use for training ('cpu' or 'cuda').
        Lambda (float): Weight for the physical residual in the loss function.
        max_phy_loss (float): Maximum physical loss at epoch 1

    Returns:
        list: Validation losses across all folds.
        nn.Module: The trained model.
    """
    writer = SummaryWriter()
    model.to(device)
    data = [dat.to(device) for dat in data]

    kf = KFold(n_splits=k, shuffle=True, random_state=3407)
    validation_losses = []

    global_best_model_state = None
    global_best_loss = float('inf')

    for fold, (train_index, val_index) in enumerate(kf.split(data), 1):
        print(f"\nStarting fold {fold}/{k}...")

        if global_best_model_state is not None:
            model.load_state_dict(global_best_model_state) 

        train_data = [data[i] for i in train_index]
        val_data = [data[i] for i in val_index]


        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)


        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
        criterion = PenalizedSmoothL1Loss(penalty_weight=penalty_weight)

        # Adjust the scheduler's patience if needed
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=300, factor=0.5)

        LAMBDA = (Lambda + (1 - Lambda) * np.exp(- np.log(1/Lambda) * (fold-1) / (k-1))) * max_phy_loss

        def train():
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                # Add noise to input features except the first column
                noisy_x = batch.x.clone()
                noisy_x[:, 1:] += noise_level * torch.randn_like(batch.x[:, 1:])
                batch.x = noisy_x
                out = model(batch)
                # compute the data loss
                loss = criterion(out, batch.y)

                # Compute the physical residual and physical loss
                physical_loss = compute_physical_residual(out, batch)

                total_batch_loss = loss + LAMBDA * physical_loss

                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                total_loss += total_batch_loss.item()
            # Adjust the scheduler after each epoch
            scheduler.step(total_loss / len(train_loader))
            return total_loss / len(train_loader)

        best_loss = float('inf')
        patience = 500
        counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = train()
            
            # Compute validation loss
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    data_loss = criterion(out, batch.y)
                    physical_loss = compute_physical_residual(out, batch)
                    
                    total_batch_loss = data_loss + LAMBDA * physical_loss
                    val_loss += total_batch_loss.item()
            val_loss /= len(val_loader)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, \
                      Validation Physical Loss: {physical_loss:.4f}')

            writer.add_scalar(f'Fold_{fold}/Loss/train', train_loss, epoch)
            writer.add_scalar(f'Fold_{fold}/Loss/val', val_loss, epoch)

            # Early stopping based on validation loss
            
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                global_best_model_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch}. Best validation loss: {best_loss:.4f}')
                    break


            
        # After training, load the best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save validation loss for this fold
        validation_losses.append(best_loss)
        print(f'Fold {fold} completed. Best validation loss: {best_loss:.4f}')

    avg_val_loss = sum(validation_losses) / k
    print(f'\nTraining completed. Average validation loss across folds: {avg_val_loss:.4f}')

    # tensorboard writer
    writer.flush()
    writer.close()

    return validation_losses, model, avg_val_loss

