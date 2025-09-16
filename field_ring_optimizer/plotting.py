# field_ring_optimizer/plotting.py

import matplotlib.pyplot as plt
import os

def visualize_progress(epoch, history, vertices, results_dir):
    """Saves a plot of the current geometry and the loss curve."""
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Geometry
    vertices_np = vertices.cpu().detach().numpy()
    ax1.plot(vertices_np[:, 0], vertices_np[:, 1], 'ko-', markersize=3)
    ax1.fill(vertices_np[:, 0], vertices_np[:, 1], 'c', alpha=0.3)
    ax1.set_title(f"Structure at Epoch {epoch}")
    ax1.set_xlabel("x (μm)")
    ax1.set_ylabel("y (μm)")
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot 2: Loss Curve
    ax2.semilogy(history['epoch'], history['task_loss'], label='Task Loss (Weighted & Normalized)')
    ax2.set_title("Loss Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Log Loss")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"progress_epoch_{epoch}.png"))
    plt.close(fig)

def plot_parameter_history(history, params, results_dir):
    """Saves plots of the geometry parameters (s and w) over epochs."""
    plt.ioff()
    s_keys = sorted([k for k in params.keys() if k.startswith('s')])
    w_keys = sorted([k for k in params.keys() if k.startswith('w')])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    for key in s_keys:
        ax1.plot(history['epoch'], history[key], label=key)
    ax1.set_title('s Parameters vs. Epochs')
    ax1.set_ylabel('Value (μm)')
    ax1.grid(True)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    for key in w_keys:
        ax2.plot(history['epoch'], history[key], label=key)
    ax2.set_title('w Parameters vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value (μm)')
    ax2.grid(True)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(results_dir, "parameter_history.png"))
    plt.close(fig)