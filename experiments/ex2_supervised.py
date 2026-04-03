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
# 3. Main (Modified for strict compliance)
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

    # ---------------------------------------------------------
    # 【关键修改】：在循环外部生成“固定”的训练集和测试集
    # ---------------------------------------------------------
    N_data_train = 50000  # 固定的训练集大小
    N_data_val = 10000    # 固定的验证集大小

    print(f"Generating fixed training dataset of size {N_data_train}...")
    train_inputs, train_v_targets, train_a_targets = generate_training_batch(lqr_solver, N_data_train, T_val, device)
    
    print(f"Generating fixed validation dataset of size {N_data_val}...")
    val_inputs, val_v_targets, val_a_targets = generate_training_batch(lqr_solver, N_data_val, T_val, device)

    # 初始化网络和优化器
    model_v = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    model_a = NetFFN(input_dim=3, hidden_dim=100, output_dim=2).to(device)

    # 学习率可以稍微调大一点，因为我们是在整个大 batch 上做全梯度下降 (Full-batch Gradient Descent)
    optimizer_v = optim.Adam(model_v.parameters(), lr=5e-3)
    optimizer_a = optim.Adam(model_a.parameters(), lr=5e-3)

    scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=1000, gamma=0.5)
    scheduler_a = optim.lr_scheduler.StepLR(optimizer_a, step_size=1000, gamma=0.5)

    criterion = nn.MSELoss()
    epochs = 4000

    loss_history_v = []
    loss_history_a = []
    val_loss_history_v = []
    val_loss_history_a = []

    print(f"Starting training for {epochs} epochs on fixed dataset...")

    for epoch in range(epochs):
        model_v.train()
        model_a.train()

        # Train value network (Ex 2.1)
        optimizer_v.zero_grad()
        v_pred = model_v(train_inputs)
        loss_v = criterion(v_pred, train_v_targets)
        loss_v.backward()
        optimizer_v.step()
        scheduler_v.step()

        # Train action network (Ex 2.2)
        optimizer_a.zero_grad()
        a_pred = model_a(train_inputs)
        loss_a = criterion(a_pred, train_a_targets)
        loss_a.backward()
        optimizer_a.step()
        scheduler_a.step()

        loss_history_v.append(loss_v.item())
        loss_history_a.append(loss_a.item())

        # Validation step
        model_v.eval()
        model_a.eval()
        with torch.no_grad():
            val_v_pred = model_v(val_inputs)
            val_a_pred = model_a(val_inputs)
            val_v = criterion(val_v_pred, val_v_targets).item()
            val_a = criterion(val_a_pred, val_a_targets).item()
            
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

    # 画图部分保持你的原本优秀逻辑不变
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history_v, label="Training Loss", alpha=0.9)
    plt.plot(val_loss_history_v, label="Validation Loss", alpha=0.9)
    plt.yscale("log")
    plt.title("Exercise 2.1: Value Function Training Loss (NetDGM)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.subplot(1, 2, 2)
    plt.plot(loss_history_a, label="Training Loss", alpha=0.9)
    plt.plot(val_loss_history_a, label="Validation Loss", alpha=0.9)
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