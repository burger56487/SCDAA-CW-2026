import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from ex1_lqr_mc import LQR_Solver


# ==========================================
# 0. Reproducibility
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 1. Network Architectures
# ==========================================
class NetDGM(nn.Module):
    """
    Exercise 2.1:
    One hidden layer of width 100 for approximating v(t, x).
    Input dimension = 3  (t, x1, x2)
    Output dimension = 1
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
    Exercise 2.2:
    Two hidden layers of width 100 for approximating a(t, x).
    Input dimension = 3  (t, x1, x2)
    Output dimension = 2
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
# 2. Data Generation
# ==========================================
def generate_training_batch(lqr_solver, batch_size, T, device):
    """
    Sample:
        t ~ Uniform([0, T])
        x ~ Uniform([-3, 3]^2)

    Returns:
        inputs : shape (batch_size, 3) with columns [t, x1, x2]
        v_true : shape (batch_size, 1)
        a_true : shape (batch_size, 2)
    """
    t_batch = torch.rand(batch_size, dtype=torch.float32, device=device) * T
    x_batch_flat = torch.rand((batch_size, 2), dtype=torch.float32, device=device) * 6.0 - 3.0
    x_batch = x_batch_flat.unsqueeze(1)  # (batch, 1, 2)

    inputs = torch.cat([t_batch.unsqueeze(1), x_batch_flat], dim=1)

    with torch.no_grad():
        v_true = lqr_solver.value_function(t_batch, x_batch)   # (batch, 1)
        a_true = lqr_solver.markov_control(t_batch, x_batch)   # (batch, 2)

    return inputs, v_true, a_true


@torch.no_grad()
def evaluate_models(model_v, model_a, lqr_solver, batch_size, T, device, criterion):
    model_v.eval()
    model_a.eval()

    inputs, v_true, a_true = generate_training_batch(lqr_solver, batch_size, T, device)
    v_pred = model_v(inputs)
    a_pred = model_a(inputs)

    val_loss_v = criterion(v_pred, v_true).item()
    val_loss_a = criterion(a_pred, a_true).item()

    model_v.train()
    model_a.train()

    return val_loss_v, val_loss_a


# ==========================================
# 3. Main
# ==========================================
if __name__ == "__main__":
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    # Problem setup (T = 1 required by coursework)
    H_mat = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
    M_mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sigma_mat = np.array([[0.2, 0.0], [0.0, 0.2]], dtype=np.float64)
    C_mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    D_mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    T_val = 1.0

    print("Initializing LQR benchmark...")
    lqr_solver = LQR_Solver(H_mat, M_mat, sigma_mat, C_mat, D_mat, R_mat, T_val)
    time_grid = np.linspace(0.0, T_val, 500)
    lqr_solver.solve_riccati(time_grid)
    print("LQR benchmark ready.")

    model_v = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    model_a = NetFFN(input_dim=3, hidden_dim=100, output_dim=2).to(device)

    optimizer_v = optim.Adam(model_v.parameters(), lr=2e-3)
    optimizer_a = optim.Adam(model_a.parameters(), lr=2e-3)

    scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=1000, gamma=0.5)
    scheduler_a = optim.lr_scheduler.StepLR(optimizer_a, step_size=1000, gamma=0.5)

    criterion = nn.MSELoss()

    epochs = 3000
    batch_size = 4096
    eval_batch_size = 4096

    loss_history_v = []
    loss_history_a = []
    val_loss_history_v = []
    val_loss_history_a = []

    print(f"Starting training for {epochs} epochs...")

    model_v.train()
    model_a.train()

    for epoch in range(epochs):
        inputs, v_targets, a_targets = generate_training_batch(
            lqr_solver=lqr_solver,
            batch_size=batch_size,
            T=T_val,
            device=device
        )

        # Train value network
        optimizer_v.zero_grad()
        v_pred = model_v(inputs)
        loss_v = criterion(v_pred, v_targets)
        loss_v.backward()
        optimizer_v.step()
        scheduler_v.step()

        # Train action network
        optimizer_a.zero_grad()
        a_pred = model_a(inputs)
        loss_a = criterion(a_pred, a_targets)
        loss_a.backward()
        optimizer_a.step()
        scheduler_a.step()

        loss_history_v.append(loss_v.item())
        loss_history_a.append(loss_a.item())

        val_v, val_a = evaluate_models(
            model_v=model_v,
            model_a=model_a,
            lqr_solver=lqr_solver,
            batch_size=eval_batch_size,
            T=T_val,
            device=device,
            criterion=criterion
        )
        val_loss_history_v.append(val_v)
        val_loss_history_a.append(val_a)

        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch [{epoch + 1:4d}/{epochs}] | "
                f"Train Loss V: {loss_v.item():.6e} | "
                f"Train Loss A: {loss_a.item():.6e} | "
                f"Eval Loss V: {val_v:.6e} | "
                f"Eval Loss A: {val_a:.6e}"
            )

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history_v, label="Training Loss", alpha=0.9)
    plt.plot(val_loss_history_v, label="Fresh-batch Eval Loss", alpha=0.9)
    plt.yscale("log")
    plt.title("Exercise 2.1: Value Function Training Loss (NetDGM)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.subplot(1, 2, 2)
    plt.plot(loss_history_a, label="Training Loss", alpha=0.9)
    plt.plot(val_loss_history_a, label="Fresh-batch Eval Loss", alpha=0.9)
    plt.yscale("log")
    plt.title("Exercise 2.2: Markov Control Training Loss (NetFFN)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()

    save_path = os.path.join("plots", "ex2_supervised_loss.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved to '{save_path}'.")

    print("\nSuggested table entries every 500 epochs:")
    print("Epoch | Loss V (train) | Loss A (train) | Loss V (eval) | Loss A (eval)")
    for k in [500, 1000, 1500, 2000, 2500, 3000]:
        idx = k - 1
        print(
            f"{k:5d} | "
            f"{loss_history_v[idx]:.6e} | "
            f"{loss_history_a[idx]:.6e} | "
            f"{val_loss_history_v[idx]:.6e} | "
            f"{val_loss_history_a[idx]:.6e}"
        )