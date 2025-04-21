# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# --- Import freeze_support ---
from multiprocessing import freeze_support
# --------------------------
from model import TransformerSimilarityModel # Assuming model.py exists and is correct
from data import build_dataloaders # Import the modified function
import matplotlib.pyplot as plt

# --- Evaluation Function (Keep outside the main block) ---
def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the given dataloader."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculations
        for sample1, sample2, labels in dataloader:
            sample1 = sample1.to(device)
            sample2 = sample2.to(device)
            labels = labels.float().unsqueeze(1).to(device) # Ensure correct shape and type

            outputs = model(sample1, sample2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs)) # Get predictions (0 or 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy
# ---------------------------------------------------------

# --- Main execution block ---
if __name__ == '__main__':
    # --- Add freeze_support() here ---
    freeze_support()
    # ---------------------------------

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    NUM_EPOCHS = 200 # Reduced for quicker testing, adjust as needed
    BATCH_SIZE = 32 # Reduced for potentially smaller dataset after split
    LEARNING_RATE = 1e-4 # Adjusted learning rate
    MAX_FRAMES = 100 # Keep consistent with data loading
    INPUT_DIM = 63 # Derived from data processing (64 - 1)
    MODEL_DIM = 128 # Example: Adjust based on model complexity needs
    NUM_HEADS = 8   # Example: Adjust based on model complexity needs
    NUM_LAYERS = 2  # Example: Adjust based on model complexity needs
    CHECKPOINT_SAVE_INTERVAL = 10 # Save every 10 epochs
    LR_SCHEDULER_STEP = 10
    LR_SCHEDULER_GAMMA = 0.7
    NUM_WORKERS = 2 # Set number of workers for DataLoader


    # --- Data Loading ---
    DATA_DIR = r'Hand_pose_annotation_v1\New_Hand_pose_annotation_v1_lines_70_130' # MAKE SURE THIS PATH IS CORRECT
    print("Building dataloaders...")
    try:
        train_loader, test_loader = build_dataloaders(
            data_dir=DATA_DIR,
            max_frames=MAX_FRAMES,
            batch_size=BATCH_SIZE,
            test_split=0.2, # 80% train, 20% test
            # Pass num_workers to the dataloader function if it accepts it,
            # otherwise it's handled during DataLoader creation inside build_dataloaders
            # num_workers=NUM_WORKERS # Modify build_dataloaders if needed
        )
        # Make sure build_dataloaders uses NUM_WORKERS when creating DataLoaders
        # Example modification in data.py's build_dataloaders:
        # train_loader = DataLoader(..., num_workers=num_workers, ...)
        # test_loader = DataLoader(..., num_workers=num_workers, ...)
        print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except FileNotFoundError:
        print(f"Error: Data directory not found at {DATA_DIR}")
        print("Please ensure the DATA_DIR path is correct.")
        exit()
    except ValueError as ve:
         print(f"Error building dataloaders: {ve}")
         print("Please check data structure and pairing logic in data.py.")
         exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        exit() # Exit if data loading fails
    # --------------------


    # Initialize Model, Loss, Optimizer, Scheduler
    print("Initializing model...")
    # Ensure model definition is compatible with hyperparameters
    try:
        model = TransformerSimilarityModel(
             input_dim=INPUT_DIM,
             model_dim=MODEL_DIM,
             num_heads=NUM_HEADS,
             num_layers=NUM_LAYERS
        ).to(device)
    except NameError:
         print("Error: 'TransformerSimilarityModel' not found. Make sure 'model.py' exists and the class is defined correctly.")
         exit()
    except Exception as e:
         print(f"Error initializing model: {e}")
         exit()

    criterion = nn.BCEWithLogitsLoss() # Suitable for binary classification with logits output
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)
    print("Model, criterion, optimizer, scheduler initialized.")

    # Logging and Checkpoints
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"runs/experiment_{current_time}"
    checkpoint_dir = f"checkpoints/experiment_{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"Logging to: {log_dir}")
    print(f"Checkpoints will be saved in: {checkpoint_dir}")


    # Lists for plotting training curves
    epoch_train_losses = []
    epoch_train_accuracies = []
    # Optional: Store test metrics if evaluating periodically
    epoch_test_losses = []
    epoch_test_accuracies = []
    evaluated_epochs = []


    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, batch_data in enumerate(train_loader):
            try:
                sample1, sample2, labels = batch_data
            except ValueError as e:
                 print(f"Error unpacking batch {i} in epoch {epoch+1}: {e}")
                 print(f"Batch data type: {type(batch_data)}")
                 if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                     print(f"Batch data length: {len(batch_data)}")
                     # Potentially print shapes of elements if they exist
                 continue # Skip this problematic batch


            sample1, sample2 = sample1.to(device), sample2.to(device)
            labels = labels.float().unsqueeze(1).to(device) # Target shape [batch_size, 1]

            # Forward pass
            outputs = model(sample1, sample2)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Print batch info occasionally
            if (i + 1) % 20 == 0: # Print every 20 batches
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')

        # Calculate metrics for the epoch
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_acc = 100 * correct_train / total_train if total_train > 0 else 0

        epoch_train_losses.append(epoch_loss)
        epoch_train_accuracies.append(epoch_acc)

        # Log to TensorBoard
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/accuracy', epoch_acc, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS} Summary: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')

        # --- Optional: Evaluate on Test Set Periodically (e.g., every 5 epochs) ---
        test_loss_epoch, test_acc_epoch = None, None # Reset for this epoch
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            test_loss_epoch, test_acc_epoch = evaluate(model, test_loader, criterion, device)
            writer.add_scalar('test/loss', test_loss_epoch, epoch)
            writer.add_scalar('test/accuracy', test_acc_epoch, epoch)
            # Store for plotting
            epoch_test_losses.append(test_loss_epoch)
            epoch_test_accuracies.append(test_acc_epoch)
            evaluated_epochs.append(epoch + 1) # Store epoch number
            print(f'Epoch {epoch+1}/{NUM_EPOCHS} Evaluation: Test Loss: {test_loss_epoch:.4f}, Test Acc: {test_acc_epoch:.2f}%')
        # -----------------------------------------------------------------------

        # Learning rate scheduling
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_acc,
                'test_loss': test_loss_epoch, # Save test loss if evaluated
                'test_accuracy': test_acc_epoch # Save test acc if evaluated
            }
            torch.save(save_dict, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


    # --- Final Evaluation on Test Set ---
    print("\n--- Training Complete ---")
    print("Evaluating on Test Set...")
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n--- Final Test Results ---")
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_acc:.2f}%")
    # ---------------------------------

    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    writer.close()

    # --- Plotting Training Curves ---
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, epoch_train_losses, label='Training Loss', color='blue', marker='.')
    # Plot test loss only for epochs where it was evaluated
    if evaluated_epochs:
         plt.plot(evaluated_epochs, epoch_test_losses, label='Test Loss', color='red', linestyle='--', marker='x')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, epoch_train_accuracies, label='Training Accuracy', color='blue', marker='.')
    # Plot test accuracy only for epochs where it was evaluated
    if evaluated_epochs:
         plt.plot(evaluated_epochs, epoch_test_accuracies, label='Test Accuracy', color='red', linestyle='--', marker='x')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.suptitle('Training and Test Metrics Over Epochs')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_save_path = os.path.join(log_dir, 'training_curves.png')
    plt.savefig(plot_save_path)
    print(f"Training curves plot saved to {plot_save_path}")
    plt.show()
    # --------------------------------

    print("Script finished.")