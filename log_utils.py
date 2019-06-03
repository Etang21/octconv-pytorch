""" 
Utility functions for plotting our network training logs 

Assumes logs are formatted like:
Epoch: 99, train Loss: 0.9492 Acc: 0.7713
Epoch: 99, val Loss: 1.7540 Acc: 0.6172

"""


def get_history(logfile):
    """Returns a dictionary with lists:
    
    train_losses: list of training losses
    val_losses: ditto
    train_accuracies
    val_accuracies
    """
    stats = {"train_losses": [],
             "val_losses": [],
             "train_accuracies": [],
             "val_accuracies": []
            }
    with open(logfile, "r") as f:
        for line in f:
            tokens = line.split()
            if "Epoch:" not in tokens:
                continue
            if "train" in tokens:
                stats["train_losses"].append(float(tokens[4]))
                stats["train_accuracies"].append(float(tokens[6]))
            elif "val" in tokens:
                stats["val_losses"].append(float(tokens[4]))
                stats["val_accuracies"].append(float(tokens[6]))
    return stats