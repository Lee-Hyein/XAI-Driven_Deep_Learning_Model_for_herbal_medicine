import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# # Configure matplotlib fonts to support Korean text
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['axes.unicode_minus'] = False

class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super(VGG16Classifier, self).__init__()

        # Load the pretrained VGG16 backbone
        self.vgg16 = models.vgg16(pretrained=pretrained)

        # Optionally freeze the convolutional backbone
        if freeze_backbone:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False

        # Replace the classifier head with task-specific layers
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.vgg16(x)

def get_transforms(img_size=224):
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    return transform_train, transform_val

# Helper used to run each of the three repeated experiments
def save_cnn_checkpoint(model, optimizer, epoch, metric, filepath, experiment_idx):
    """Persist CNN weights and metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': metric,
        'experiment': experiment_idx
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def run_single_experiment(experiment_id, device, train_loader, val_loader, results_dir, checkpoint_root, args):
    """Run a single experiment"""
    print(f"\n=== 실험 {experiment_id + 1}/3 시작 ===")
    
    # Initialize a fresh model instance for every experiment
    model = VGG16Classifier(num_classes=args.num_classes, pretrained=True, freeze_backbone=False)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    
    # Prepare per-experiment checkpoint directory
    exp_checkpoint_dir = os.path.join(checkpoint_root, f"experiment_{experiment_id + 1}")
    os.makedirs(exp_checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(exp_checkpoint_dir, 'best_model.pth')
    last_checkpoint_path = os.path.join(exp_checkpoint_dir, 'last_model.pth')
    
    # Containers for per-epoch metrics collected within this experiment
    epochs = []
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    epoch_times = []
    total_times = []
    
    train_epoch = args.epochs
    total_start_time = time.time()
    best_val_acc = -1.0
    
    for epoch in range(train_epoch):
        # Measure epoch start time for throughput tracking
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.2f}%")
        
        # Track training accuracy and loss for this epoch
        train_acc = 100. * correct / total
        train_loss = running_loss / total  # Average loss per epoch
        train_accs.append(train_acc)
        train_losses.append(train_loss)  # Store loss value

        scheduler.step()

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Compute epoch duration and cumulative training time
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        
        # Track validation accuracy and loss for this epoch
        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_loss / val_total  # Average loss per epoch
        val_accs.append(val_acc)
        val_losses.append(val_loss_avg)  # Store loss value
        epochs.append(epoch + 1)
        epoch_times.append(epoch_time)  # Store epoch duration
        total_times.append(total_time)  # Store cumulative training time
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_cnn_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metric=val_acc,
                filepath=best_checkpoint_path,
                experiment_idx=experiment_id + 1
            )
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%, Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s')
    
    save_cnn_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=train_epoch,
        metric=best_val_acc,
        filepath=last_checkpoint_path,
        experiment_idx=experiment_id + 1
    )
    
    # Return metrics captured in this experiment
    return {
        'epochs': epochs,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch_times': epoch_times,
        'total_times': total_times
    }

def calculate_std_deviation(experiments_data):
    """Compute standard deviation for experiment metrics"""
    num_experiments = len(experiments_data)
    num_epochs = len(experiments_data[0]['train_accs'])
    
    # Compute standard deviation for every epoch across experiments
    train_acc_std = []
    val_acc_std = []
    train_loss_std = []
    val_loss_std = []
    
    for epoch_idx in range(num_epochs):
        # Gather results for the same epoch across experiments
        epoch_train_accs = [exp['train_accs'][epoch_idx] for exp in experiments_data]
        epoch_val_accs = [exp['val_accs'][epoch_idx] for exp in experiments_data]
        epoch_train_losses = [exp['train_losses'][epoch_idx] for exp in experiments_data]
        epoch_val_losses = [exp['val_losses'][epoch_idx] for exp in experiments_data]
        
        # Calculate the standard deviation
        train_acc_std.append(np.std(epoch_train_accs))
        val_acc_std.append(np.std(epoch_val_accs))
        train_loss_std.append(np.std(epoch_train_losses))
        val_loss_std.append(np.std(epoch_val_losses))
    
    return train_acc_std, val_acc_std, train_loss_std, val_loss_std

# Entry point
if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CNN model training')
    parser.add_argument('--train_dir', type=str, 
                       default='/mnt/mydisk/hyein/TransFG_data/dataset_cls_PE_aug/train',
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str,
                       default='/mnt/mydisk/hyein/TransFG_data/dataset_cls_PE_aug/val',
                       help='Validation data directory (optional)')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--results_dir', type=str,
                       default='/mnt/mydisk/hyein/TransFG_data/model/CNN/results',
                       help='Results storage directory')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='/mnt/mydisk/hyein/TransFG_data/model/CNN/checkpoints',
                       help='Model checkpoint directory')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training data: {args.train_dir}")
    print(f"Validation data: {args.val_dir}")
    print(f"Number of classes: {args.num_classes}")
    
    # Configure the dataloaders
    transform_train, transform_val = get_transforms(224)
    train_dataset = datasets.ImageFolder(args.train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create result directories as needed
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Run the three repeated experiments
    experiments_data = []
    for exp_id in range(3):
        # Use different random seeds per experiment
        torch.manual_seed(42 + exp_id)
        np.random.seed(42 + exp_id)
        
        experiment_result = run_single_experiment(exp_id, device, train_loader, val_loader, results_dir, args.checkpoint_dir, args)
        experiments_data.append(experiment_result)
    
    # Calculate the standard deviation
    train_acc_std, val_acc_std, train_loss_std, val_loss_std = calculate_std_deviation(experiments_data)
    
    # Use the first experiment as the representative curve (mean proxy)
    epochs = experiments_data[0]['epochs']
    train_accs = experiments_data[0]['train_accs']
    val_accs = experiments_data[0]['val_accs']
    train_losses = experiments_data[0]['train_losses']
    val_losses = experiments_data[0]['val_losses']
    epoch_times = experiments_data[0]['epoch_times']
    total_times = experiments_data[0]['total_times']
    
    # Plot accuracy and loss curves with error bars
    plt.figure(figsize=(15, 10))
    
    # Accuracy plot (real standard deviation)
    plt.subplot(2, 1, 1)
    plt.errorbar(epochs, train_accs, yerr=train_acc_std, fmt='b-o', 
                label='Training Accuracy', capsize=3, capthick=1)
    plt.errorbar(epochs, val_accs, yerr=val_acc_std, fmt='r-o', 
                label='Validation Accuracy', capsize=3, capthick=1)
    plt.title('Accuracy by Epoch (Actual Standard Deviation Error Bar)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Loss plot (real standard deviation)
    plt.subplot(2, 1, 2)
    plt.errorbar(epochs, train_losses, yerr=train_loss_std, fmt='b-o', 
                label='Training Loss', capsize=3, capthick=1)
    plt.errorbar(epochs, val_losses, yerr=val_loss_std, fmt='r-o', 
                label='Validation Loss', capsize=3, capthick=1)
    plt.title('Loss by Epoch (Actual Standard Deviation Error Bar)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot to disk
    plt.savefig(os.path.join(results_dir, 'training_results.png'), dpi=300)
    plt.show()
    
    # Report total training time using the first experiment
    total_training_time = experiments_data[0]['total_times'][-1]  # Total time recorded at the last epoch
    print(f"\nTotal training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Average epoch time: {total_training_time/args.epochs:.2f} seconds")

    # Export metrics (including true std) for Origin
    results_df = pd.DataFrame({
        'Epoch': epochs,
        'Train_Loss': train_losses,
        'Train_Accuracy': train_accs,
        'Val_Loss': val_losses,
        'Val_Accuracy': val_accs,
        'Epoch_Time': epoch_times,
        'Total_Time': total_times,
        'Train_Acc_ErrorBar': train_acc_std,  # Actual standard deviation
        'Val_Acc_ErrorBar': val_acc_std,      # Actual standard deviation
        'Train_Loss_ErrorBar': train_loss_std, # Actual standard deviation
        'Val_Loss_ErrorBar': val_loss_std     # Actual standard deviation
    })
    
    csv_path = os.path.join(results_dir, 'training_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'Training results saved to CSV file for Origin program: {csv_path}')
    print(f'Actual standard deviation calculated and used in Error Bar.')
