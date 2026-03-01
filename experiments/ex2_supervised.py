import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import your implemented LQR_Solver
from ex1_lqr_mc import LQR_Solver

# ==========================================
# 1. Network Architectures Definition
# ==========================================
class NetDGM(nn.Module):
    """
    Exercise 2.1: Single hidden layer network to approximate the value function v(t,x).
    Contains 1 hidden layer of size 100 as required.
    """
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super(NetDGM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class NetFFN(nn.Module):
    """
    Exercise 2.2: Double hidden layer network to approximate the Markov control a(t,x).
    Contains 2 hidden layers of size 100, output dimension is 2.
    """
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super(NetFFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Dynamic Data Generation Logic
# ==========================================
def generate_training_batch(lqr_solver, batch_size, T, device):
    """
    Dynamically generates random batch data at each epoch to prevent overfitting 
    and improve generalization.
    """
    # Sample t uniformly from [0, T]
    t_batch = torch.rand(batch_size, dtype=torch.float32, device=device) * T
    
    # Sample x uniformly from [-3, 3] x [-3, 3]
    x_batch_flat = torch.rand((batch_size, 2), dtype=torch.float32, device=device) * 6.0 - 3.0
    x_batch = x_batch_flat.unsqueeze(1) # shape: (batch_size, 1, 2) to match LQR_Solver
    
    # Concatenate to form network input features (t, x_1, x_2) -> shape: (batch_size, 3)
    inputs = torch.cat([t_batch.unsqueeze(1), x_batch_flat], dim=1)
    
    # Compute ground truth without tracking gradients
    with torch.no_grad():
        v_true = lqr_solver.value_function(t_batch, x_batch)    # Output shape: (batch_size, 1)
        a_true = lqr_solver.markov_control(t_batch, x_batch)    # Output shape: (batch_size, 2)
        
    return inputs, v_true, a_true

# ==========================================
# 3. Main Training Workflow
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current computation device: {device}")

    # ---------------- Configure and Solve LQR ----------------
    H_mat = np.array([[0.1, 0.0], [0.0, 0.1]])
    M_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma_mat = np.array([[0.2, 0.0], [0.0, 0.2]])
    C_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    D_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    T_val = 1.0

    print("Initializing base LQR solver to provide ground truth labels...")
    lqr_solver = LQR_Solver(H_mat, M_mat, sigma_mat, C_mat, D_mat, R_mat, T_val)
    time_grid = np.linspace(0, T_val, 500)
    lqr_solver.solve_riccati(time_grid)

    # ---------------- Instantiate Networks and Optimizers ----------------
    model_v = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    model_a = NetFFN(input_dim=3, hidden_dim=100, output_dim=2).to(device) 
    
    # Use Torch Adam Optimizer
    optimizer_v = optim.Adam(model_v.parameters(), lr=2e-3)
    optimizer_a = optim.Adam(model_a.parameters(), lr=2e-3)
    
    # Optimization: Add Learning Rate Schedulers for smoother convergence
    scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=1000, gamma=0.5)
    scheduler_a = optim.lr_scheduler.StepLR(optimizer_a, step_size=1000, gamma=0.5)
    
    criterion = nn.MSELoss()

    # ---------------- Training Loop ----------------
    epochs = 3000
    batch_size = 4096  # Sample 4096 points per epoch
    loss_history_v = []
    loss_history_a = []
    
    print(f"Starting parallel training for both networks ({epochs} Epochs, {batch_size} points dynamically sampled per epoch)...")
    
    # Explicitly set models to training mode
    model_v.train()
    model_a.train()
    
    for epoch in range(epochs):
        # Dynamically generate batch data for current epoch
        inputs, v_targets, a_targets = generate_training_batch(lqr_solver, batch_size, T_val, device)

        # Train Value Function Network v(t,x)
        optimizer_v.zero_grad()
        v_pred = model_v(inputs)
        loss_v = criterion(v_pred, v_targets)
        loss_v.backward()
        optimizer_v.step()
        scheduler_v.step()
        loss_history_v.append(loss_v.item())
        
        # Train Markov Control Network a(t,x)
        optimizer_a.zero_grad()
        a_pred = model_a(inputs)
        loss_a = criterion(a_pred, a_targets)
        loss_a.backward()
        optimizer_a.step()
        scheduler_a.step()
        loss_history_a.append(loss_a.item())
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch+1:4d}/{epochs}] | Loss V: {loss_v.item():.6e} | Loss A: {loss_a.item():.6e}")

    # ---------------- Plot and Save Loss Charts ----------------
    plt.figure(figsize=(14, 6))

    # Plot 1: Value Function Training Loss (Exercise 2.1)
    plt.subplot(1, 2, 1)
    plt.plot(loss_history_v, color='blue', alpha=0.8, label='Value Function Loss (MSE)')
    plt.yscale('log')
    plt.title('Exercise 2.1: Training Loss for Value Function (NetDGM)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Plot 2: Markov Control Training Loss (Exercise 2.2)
    plt.subplot(1, 2, 2)
    plt.plot(loss_history_a, color='red', alpha=0.8, label='Markov Control Loss (MSE)')
    plt.yscale('log')
    plt.title('Exercise 2.2: Training Loss for Markov Control (NetFFN)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('ex2_supervised_loss.png', dpi=300)
    print("\n✅ Training complete! The loss plots have been saved as 'ex2_supervised_loss.png'.")