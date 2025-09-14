#!/usr/bin/env python3
import re
import sys

def parse_log(file_path, keyword):
    """
    Extract epochs and losses from log file lines containing the given keyword.
    Returns two lists: epochs and losses.
    """
    epochs = []
    losses = []

    with open(file_path, "r") as f:
        for line in f:
            if keyword in line:
                # Expecting format like: ...,E:<number>,...,L:<number>,...
                parts = line.strip().split(',')
                e_match = re.search(r"E:([0-9]+)", parts[1])
                l_match = re.search(r"L:([0-9.]+)", parts[3])
                if e_match and l_match:
                    epochs.append(int(e_match.group(1)))
                    losses.append(float(l_match.group(1)))
    return epochs, losses

def print_loss_data(log_file):
    """Print loss data in a readable format without requiring matplotlib"""
    
    # Parse training and validation logs
    train_epochs, train_losses = parse_log(log_file, "Train")
    valid_epochs, valid_losses = parse_log(log_file, "Valid")
    
    print("=== SSD Training Loss Analysis ===")
    print()
    
    if train_epochs:
        print("Training Loss:")
        print("Epoch\tLoss")
        print("-" * 20)
        for epoch, loss in zip(train_epochs, train_losses):
            print(f"{epoch}\t{loss:.3f}")
        
        print()
        print(f"Training Summary:")
        print(f"  Initial Loss (Epoch 1): {train_losses[0]:.3f}")
        print(f"  Final Loss (Epoch {train_epochs[-1]}): {train_losses[-1]:.3f}")
        print(f"  Total Reduction: {train_losses[0] - train_losses[-1]:.3f}")
        print(f"  Improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    
    print()
    
    if valid_epochs:
        print("Validation Loss:")
        print("Epoch\tLoss")
        print("-" * 20)
        for epoch, loss in zip(valid_epochs, valid_losses):
            print(f"{epoch}\t{loss:.3f}")
        
        print()
        print(f"Validation Summary:")
        print(f"  Initial Loss (Epoch 1): {valid_losses[0]:.3f}")
        print(f"  Final Loss (Epoch {valid_epochs[-1]}): {valid_losses[-1]:.3f}")
        print(f"  Best Loss: {min(valid_losses):.3f} (Epoch {valid_epochs[valid_losses.index(min(valid_losses))][:1]})")
        
        # Check for overfitting
        best_epoch = valid_epochs[valid_losses.index(min(valid_losses))]
        if len(valid_losses) > 5:
            recent_avg = sum(valid_losses[-5:]) / 5
            best_loss = min(valid_losses)
            if recent_avg > best_loss * 1.1:
                print(f"  ⚠️  Possible overfitting detected after epoch {best_epoch}")
    
    print()
    
    # ASCII art plot (simple visualization)
    if train_epochs and valid_epochs:
        print("Loss Trend (ASCII Plot):")
        print("=" * 50)
        
        all_losses = train_losses + valid_losses
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        loss_range = max_loss - min_loss
        
        plot_height = 15
        
        for i in range(plot_height):
            level = max_loss - (i * loss_range / plot_height)
            line = f"{level:6.2f} |"
            
            # Plot training points
            for epoch, loss in zip(train_epochs, train_losses):
                if abs(loss - level) < loss_range / plot_height:
                    if epoch <= len(line) - 8:
                        line += " " * (epoch + 8 - len(line)) + "T"
            
            # Plot validation points  
            for epoch, loss in zip(valid_epochs, valid_losses):
                if abs(loss - level) < loss_range / plot_height:
                    if epoch <= len(line) - 8:
                        target_pos = epoch + 8
                        if target_pos >= len(line):
                            line += " " * (target_pos - len(line)) + "V"
                        elif line[target_pos] == " ":
                            line = line[:target_pos] + "V" + line[target_pos+1:]
                        else:
                            line = line[:target_pos] + "*" + line[target_pos+1:]
            
            print(line)
        
        print("       " + "-" * 20)
        print("       " + "".join([f"{i:2d}" for i in range(1, min(21, max(train_epochs[-1], valid_epochs[-1]) + 1))]))
        print()
        print("Legend: T=Training, V=Validation, *=Both")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_loss_simple.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    print_loss_data(log_file)
