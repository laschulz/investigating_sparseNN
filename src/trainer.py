from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import sys

import utils
import models
import transformer_model

def train_model(model, X_train, y_train, optimizer, loss_fn, l1_lambda=0, batch_size=32, clipping = None, device="cpu", config_path=None):
    config = utils.read_config(config_path)
    best_loss = float('inf')
    patience_counter = 0

    print(f"Using device: {device}")    
    X_train, y_train = X_train.cpu(), y_train.cpu()

    # Create a DataLoader to handle batching and shuffling
    dataset = TensorDataset(X_train, y_train)
    if device == "cuda":
        dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=8,
                persistent_workers=True,
                pin_memory=True
            )
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting training...")
    disable_tqdm = not sys.stdout.isatty()
    with tqdm(total=config["num_epochs"], desc="Training Progress", unit="epoch", disable=disable_tqdm) as pbar:
        for epoch in range(config["num_epochs"]):
            pbar.update(1)
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                y_pred = model(batch_X)
                if isinstance(model, (models.FCNN, models.FCNN_decreasing, transformer_model.NonOverlappingViT)):
                    batch_y = batch_y.view(batch_y.size(0), -1) 
                loss = loss_fn(y_pred, batch_y)

                # Apply L1 regularization if l1_lambda > 0
                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss += l1_norm * l1_lambda

                loss.backward()
                optimizer.step()

                if clipping:
                    with torch.no_grad():
                        for param in model.parameters():
                            param.data[torch.abs(param.data) < clipping] = 0.0

                epoch_loss += loss.item()  # Accumulate batch loss

            # Compute average loss for the epoch
            epoch_loss /= len(dataloader)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')

            # Early stopping logic
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping at epoch {epoch}, best loss: {best_loss:.4f}")
                    break
    return model, best_loss