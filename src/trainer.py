from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch

import utils
import models

def train_model(model, X_train, y_train, optimizer, loss_fn, l1_lambda=0, batch_size=32, clipping = None, device="cpu"):
    config = utils.read_config()
    best_loss = float('inf')
    patience_counter = 0

    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)

    # Create a DataLoader to handle batching and shuffling
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting training...")
    with tqdm(total=config["num_epochs"], desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(config["num_epochs"]):
            pbar.update(1)
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                y_pred = model(batch_X)
                if isinstance(model, models.FCNN):
                    batch_y = batch_y.view(batch_y.size(0), -1) 
                loss = loss_fn(y_pred, batch_y)

                # Apply L1 regularization if l1_lambda > 0
                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum().to(device) for p in model.parameters())
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