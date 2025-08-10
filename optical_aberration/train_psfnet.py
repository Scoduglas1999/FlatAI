import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import the custom Dataset and Model
from PSFNet_Dataset import PSFDataset
from PSFNet_Model import PSFNetCNN

def train_psfnet():
    """
    Main function to train and validate the PSFNet model.
    """
    # --- 1. HYPERPARAMETERS AND CONFIGURATION ---
    # Data Paths (using forward slashes for cross-platform compatibility)
    DATA_DIR = "C:/Users/scdou/Documents/PSFNet/training_data/"
    LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")
    IMAGES_DIR = os.path.join(DATA_DIR, "images")

    # Training Hyperparameters
    MODEL_SAVE_PATH = "./psfnet_model_final.pth"
    PLOT_SAVE_PATH = "./loss_curve.png"
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 7
    VALIDATION_SPLIT = 0.2
    
    # --- 2. DEVICE SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. DATASET AND DATALOADERS ---
    print("Loading and splitting dataset...")
    full_dataset = PSFDataset(csv_file_path=LABELS_FILE, images_dir_path=IMAGES_DIR)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create DataLoaders
    # NOTE: On Windows, setting `num_workers > 0` can cause issues with
    # multiprocessing. Setting `num_workers=0` forces data loading to occur
    # in the main process, which avoids these issues.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 4. MODEL, LOSS, AND OPTIMIZER ---
    model = PSFNetCNN(num_output_coeffs=3).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. TRAINING AND VALIDATION LOOP ---
    history = {'train_loss': [], 'val_loss': []}

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [T]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_pbar.set_postfix({'train_loss': loss.item()})

        avg_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [V]")
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                val_pbar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Validation Loss: {avg_val_loss:.6f}")

    print("\nTraining finished.")

    # --- 6. SAVE THE FINAL MODEL ---
    print(f"Saving final model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # --- 7. PLOT LOSS CURVES ---
    print(f"Generating and saving loss curve to {PLOT_SAVE_PATH}")
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    print("Plot saved successfully.")

if __name__ == '__main__':
    # To avoid issues with multiprocessing on Windows
    # it's good practice to wrap the main logic in a function
    # and call it from inside the __name__ == '__main__' block.
    train_psfnet() 