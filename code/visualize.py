import argparse
import json
import matplotlib.pyplot as plt
import os
import sys

def plot_metrics(log_files, labels, output_dir):
    # Use a nice style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        # Fallback if specific style not available
        plt.style.use('ggplot')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Data structure to hold all data
    all_data = []
    
    for i, log_path in enumerate(log_files):
        if not os.path.exists(log_path):
            print(f"Warning: File {log_path} not found. Skipping.")
            continue
            
        try:
            with open(log_path, 'r') as f:
                history = json.load(f)
                
            # Determine label
            if labels and i < len(labels):
                label = labels[i]
            else:
                # Default label from filename
                label = os.path.basename(log_path).replace('_log.json', '')
            
            data = {
                'label': label,
                'epochs': [entry['epoch'] for entry in history],
                'train_loss': [entry['train_loss'] for entry in history],
                'val_loss': [entry['val_loss'] for entry in history],
                'val_bleu': [entry['val_bleu'] for entry in history]
            }
            all_data.append(data)
        except Exception as e:
            print(f"Error reading {log_path}: {e}")

    if not all_data:
        print("No valid data found to plot.")
        return

    # Plot settings
    # Use a high-contrast colormap
    colors = plt.cm.tab10.colors 
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Helper function to save plot
    def save_plot(metric_key, title, ylabel, filename):
        plt.figure(figsize=(12, 7))
        
        for i, data in enumerate(all_data):
            plt.plot(data['epochs'], data[metric_key], 
                     label=data['label'], 
                     marker=markers[i % len(markers)], 
                     color=colors[i % len(colors)],
                     linewidth=2.5,
                     markersize=8,
                     alpha=0.8)
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"Saved {filename} to {save_path}")
        plt.close()

    # Generate plots
    save_plot('train_loss', 'Training Loss Comparison', 'Loss', 'compare_train_loss.png')
    save_plot('val_loss', 'Validation Loss Comparison', 'Loss', 'compare_val_loss.png')
    save_plot('val_bleu', 'Validation BLEU Score Comparison', 'BLEU Score', 'compare_val_bleu.png')

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare training logs")
    parser.add_argument('--log_files', type=str, nargs='+', required=True, help="List of paths to JSON log files")
    parser.add_argument('--labels', type=str, nargs='+', help="List of labels for the log files (optional)")
    parser.add_argument('--output_dir', type=str, default='plots', help="Directory to save plots")
    
    args = parser.parse_args()
    
    plot_metrics(args.log_files, args.labels, args.output_dir)

if __name__ == "__main__":
    main()
