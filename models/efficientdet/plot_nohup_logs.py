import matplotlib.pyplot as plt
import re

# Define the regex pattern to match the loss values in the log file
pattern = re.compile(r"Epoch: (\d+).*Cls loss: ([\d.]+).*Reg loss: ([\d.]+).*Total loss: ([\d.]+)")

# Initialize lists to store the epoch numbers and loss values
epoch_numbers = []
cls_losses = []
reg_losses = []
total_losses = []

# Path to the log file
log_file_path = 'nohup.out'

# Read the log file and extract the loss values
with open(log_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            cls_loss = float(match.group(2).rstrip('.'))
            reg_loss = float(match.group(3).rstrip('.'))
            total_loss = float(match.group(4).rstrip('.'))
            
            epoch_numbers.append(epoch)
            cls_losses.append(cls_loss)
            reg_losses.append(reg_loss)
            total_losses.append(total_loss)

# Save Classification Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(epoch_numbers, cls_losses, label='Cls Loss') #, marker='o')
plt.title('Classification Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cls Loss')
plt.legend()
plt.grid(True)
plt.savefig('classification_loss_plot.png')
plt.close()

# Save Regression Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(epoch_numbers, reg_losses, label='Reg Loss') #, marker='x')
plt.title('Regression Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Reg Loss')
plt.legend()
plt.grid(True)
plt.savefig('regression_loss_plot.png')
plt.close()

# Save Total Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(epoch_numbers, total_losses, label='Total Loss')#, marker='s')
plt.title('Total Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.legend()
plt.grid(True)
plt.savefig('total_loss_plot.png')
plt.close()