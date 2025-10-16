import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_training_history(filepath):
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history

def create_training_plots(history, save_path=None):
    """Create comprehensive training visualization plots."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training History Visualization', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')  # Log scale for better visibility
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss vs Accuracy scatter plot
    axes[1, 1].scatter(history['train_loss'], history['train_acc'], 
                      alpha=0.6, label='Training', s=30)
    axes[1, 1].scatter(history['val_loss'], history['val_acc'], 
                      alpha=0.6, label='Validation', s=30)
    axes[1, 1].set_title('Loss vs Accuracy Relationship', fontweight='bold')
    axes[1, 1].set_xlabel('Loss')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig

def create_detailed_analysis(history):
    """Create detailed analysis plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Training Analysis', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[0, 0].plot(epochs, loss_diff, 'purple', linewidth=2)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Validation - Training Loss (Overfitting Indicator)', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss Difference')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].fill_between(epochs, loss_diff, 0, alpha=0.3, 
                           where=(np.array(loss_diff) > 0), color='red', label='Overfitting')
    axes[0, 0].fill_between(epochs, loss_diff, 0, alpha=0.3, 
                           where=(np.array(loss_diff) <= 0), color='green', label='Good fit')
    axes[0, 0].legend()
    
    # Plot 2: Accuracy difference
    acc_diff = np.array(history['val_acc']) - np.array(history['train_acc'])
    axes[0, 1].plot(epochs, acc_diff, 'orange', linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Validation - Training Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy Difference (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Smoothed curves (moving average)
    window_size = 5
    if len(epochs) >= window_size:
        train_loss_smooth = np.convolve(history['train_loss'], np.ones(window_size)/window_size, mode='valid')
        val_loss_smooth = np.convolve(history['val_loss'], np.ones(window_size)/window_size, mode='valid')
        smooth_epochs = epochs[window_size-1:]
        
        axes[1, 0].plot(epochs, history['train_loss'], alpha=0.3, color='blue', label='Raw Training')
        axes[1, 0].plot(epochs, history['val_loss'], alpha=0.3, color='red', label='Raw Validation')
        axes[1, 0].plot(smooth_epochs, train_loss_smooth, color='blue', linewidth=2, label='Smooth Training')
        axes[1, 0].plot(smooth_epochs, val_loss_smooth, color='red', linewidth=2, label='Smooth Validation')
    else:
        axes[1, 0].plot(epochs, history['train_loss'], color='blue', linewidth=2, label='Training')
        axes[1, 0].plot(epochs, history['val_loss'], color='red', linewidth=2, label='Validation')
    
    axes[1, 0].set_title('Smoothed Loss Curves', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training metrics summary
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    best_val_acc_epoch = history['val_acc'].index(best_val_acc) + 1
    
    metrics_text = f"""Final Metrics:
    
Training Loss: {final_train_loss:.4f}
Validation Loss: {final_val_loss:.4f}
Training Accuracy: {final_train_acc:.2f}%
Validation Accuracy: {final_val_acc:.2f}%

Best Validation Accuracy: {best_val_acc:.2f}%
Best Accuracy Epoch: {best_val_acc_epoch}

Total Epochs: {len(epochs)}
Learning Rate Range: {max(history['learning_rate']):.2e} → {min(history['learning_rate']):.2e}"""
    
    axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Training Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_training_summary(history):
    """Print a text summary of the training results."""
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    epochs = len(history['train_loss'])
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    # Best metrics
    best_train_loss = min(history['train_loss'])
    best_train_loss_epoch = history['train_loss'].index(best_train_loss) + 1
    best_val_loss = min(history['val_loss'])
    best_val_loss_epoch = history['val_loss'].index(best_val_loss) + 1
    best_train_acc = max(history['train_acc'])
    best_train_acc_epoch = history['train_acc'].index(best_train_acc) + 1
    best_val_acc = max(history['val_acc'])
    best_val_acc_epoch = history['val_acc'].index(best_val_acc) + 1
    
    print(f"Total Epochs: {epochs}")
    print()
    print("FINAL METRICS:")
    print(f"  Training Loss: {final_train_loss:.4f}")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Training Accuracy: {final_train_acc:.2f}%")
    print(f"  Validation Accuracy: {final_val_acc:.2f}%")
    print()
    print("BEST METRICS:")
    print(f"  Best Training Loss: {best_train_loss:.4f} (Epoch {best_train_loss_epoch})")
    print(f"  Best Validation Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
    print(f"  Best Training Accuracy: {best_train_acc:.2f}% (Epoch {best_train_acc_epoch})")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_val_acc_epoch})")
    print()
    print("LEARNING RATE:")
    print(f"  Initial: {history['learning_rate'][0]:.2e}")
    print(f"  Final: {history['learning_rate'][-1]:.2e}")
    print()
    
    # Check for potential overfitting
    loss_diff = final_val_loss - final_train_loss
    acc_diff = final_val_acc - final_train_acc
    
    print("TRAINING ANALYSIS:")
    if loss_diff > 0.1:
        print("  ⚠️  Potential overfitting detected (validation loss > training loss)")
    elif loss_diff < -0.05:
        print("  ⚠️  Unusual pattern: training loss > validation loss")
    else:
        print("  ✅ Good training/validation loss balance")
    
    if acc_diff < -5:
        print("  ⚠️  Potential overfitting detected (training acc > validation acc)")
    elif acc_diff > 5:
        print("  ⚠️  Unusual pattern: validation acc > training acc")
    else:
        print("  ✅ Good training/validation accuracy balance")
    
    print("=" * 60)

def main():
    """Main function to run the visualization."""
    
    # Load training history
    history_path = "/home/stoneyew/Desktop/PACS/experiment/checkpoints/training_history.json"
    history = load_training_history(history_path)
    
    # Print summary
    print_training_summary(history)
    
    # Create main training plots
    print("\nGenerating main training plots...")
    main_fig = create_training_plots(history, 
                                   save_path="training_result/training_plots.png")
    
    # Create detailed analysis
    print("\nGenerating detailed analysis plots...")
    analysis_fig = create_detailed_analysis(history)
    
    print("\nVisualization complete!")
    print("Files saved:")
    print("  - training_plots.png: Main training curves")
    print("  - Detailed analysis plots displayed")

if __name__ == "__main__":
    main()
