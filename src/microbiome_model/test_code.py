import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Define the learning rate scheduler with warmup and cosine decay
class WarmupCosineDecay:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr=0.001, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr

    def lr_lambda(self, step):
        """Learning rate schedule function"""
        if step < self.warmup_steps:
            print("Warmup phase", self.base_lr)
            return 1
        else:
            decay_ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * decay_ratio))  # Cosine decay

    def get_scheduler(self):
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

# Hyperparameters
warmup_steps = 10
total_steps = 50
base_lr = 0.0001

# Initialize model and optimizer
model = SimpleModel()
optimizer = optim.AdamW(model.parameters(), lr=base_lr)

# Initialize learning rate scheduler
scheduler = WarmupCosineDecay(optimizer, warmup_steps, total_steps, base_lr).get_scheduler()
# Cosine Decay with Restarts Scheduler (Equivalent to CosineDecayRestarts in TF)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=warmup_steps,  # Number of steps before first restart
    T_mult=1,  # Multiplicative factor for decay period
    eta_min=1e-6  # Minimum learning rate
)

# Dummy dataset (random data)
X = torch.randn(100, 10)
y = torch.randn(100, 1)
criterion = nn.MSELoss()

# Track learning rates
lr_values = []

# Training loop
for step in range(total_steps):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Store learning rate for plotting
    lr_values.append(optimizer.param_groups[0]["lr"])
    
    # Print learning rate at each step
    print(f"Step {step + 1}: Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")

# Plot the learning rate schedule
plt.plot(range(total_steps), lr_values, marker="o")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Warmup + Cosine Decay")
plt.grid()
plt.show()
