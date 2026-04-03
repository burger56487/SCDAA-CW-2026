import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
# 1. Network
# ==========================================
class NetDGM(nn.Module):
    """
    DGM-style fully connected network with 3 hidden layers.
    Input dimension = 3  (t, x1, x2)
    Output dimension = 1
    """
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super(NetDGM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. Sampling Utilities
# ==========================================
def sample_interior(batch_size, T, device):
    t = torch.rand(batch_size, 1, device=device, dtype=torch.float32) * T
    x = torch.rand(batch_size, 2, device=device, dtype=torch.float32) * 6.0 - 3.0
    inputs = torch.cat([t, x], dim=1)
    inputs.requires_grad_(True)
    return inputs


def sample_terminal(batch_size, T, device):
    t = torch.full((batch_size, 1), T, device=device, dtype=torch.float32)
    x = torch.rand(batch_size, 2, device=device, dtype=torch.float32) * 6.0 - 3.0
    inputs = torch.cat([t, x], dim=1)
    return inputs


def compute_hessian_wrt_x(u, inputs):
    """
    u: shape (batch, 1)
    inputs: shape (batch, 3) = (t, x1, x2), requires_grad=True
    Returns:
        grad_all: shape (batch, 3)
        hessian_x: shape (batch, 2, 2)
    """
    grad_all = torch.autograd.grad(
        u.sum(), inputs, create_graph=True, retain_graph=True
    )[0]

    u_x = grad_all[:, 1:3]
    hessian_cols = []

    for j in range(2):
        grad2 = torch.autograd.grad(
            u_x[:, j].sum(), inputs, create_graph=True, retain_graph=True
        )[0][:, 1:3]
        hessian_cols.append(grad2.unsqueeze(-1))

    hessian_x = torch.cat(hessian_cols, dim=-1)  # (batch, 2, 2)
    return grad_all, hessian_x


# ==========================================
# 3. Monte Carlo Benchmark for Constant Control
# ==========================================
def run_mc_constant_alpha(H, M, sigma, C, D, R, alpha, x0, T, N_steps, N_samples, device):
    dt = T / N_steps
    sqrt_dt = np.sqrt(dt)

    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    M_t = torch.tensor(M, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    C_t = torch.tensor(C, dtype=torch.float32, device=device)
    D_t = torch.tensor(D, dtype=torch.float32, device=device)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)

    alpha_col = torch.tensor(alpha, dtype=torch.float32, device=device).view(2, 1)
    M_alpha = M_t @ alpha_col
    alpha_cost = (alpha_col.transpose(0, 1) @ D_t @ alpha_col).item()

    X = torch.tensor(x0, dtype=torch.float32, device=device).repeat(N_samples, 1).unsqueeze(-1)
    total_cost = torch.zeros(N_samples, device=device)

    for _ in range(N_steps):
        x_row = X.squeeze(-1)
        x_cost = torch.einsum("bi,ij,bj->b", x_row, C_t, x_row)
        total_cost += (x_cost + alpha_cost) * dt

        dW = torch.randn(N_samples, 2, 1, device=device) * sqrt_dt
        drift = torch.matmul(H_t.unsqueeze(0), X) + M_alpha.unsqueeze(0)
        diffusion = torch.matmul(sigma_t.unsqueeze(0), dW)
        X = X + drift * dt + diffusion

    x_terminal = X.squeeze(-1)
    terminal_cost = torch.einsum("bi,ij,bj->b", x_terminal, R_t, x_terminal)
    total_cost += terminal_cost

    return total_cost.mean().item()


# ==========================================
# 4. PDE Loss
# ==========================================
def dgm_losses(model, H, M, sigma, C, D, R, alpha, T, batch_interior, batch_terminal, device):
    sigma_sq = sigma @ sigma.T

    interior = sample_interior(batch_interior, T, device)
    terminal = sample_terminal(batch_terminal, T, device)

    u = model(interior)  # (batch,1)
    grad_all, hessian_x = compute_hessian_wrt_x(u, interior)

    u_t = grad_all[:, 0:1]
    u_x = grad_all[:, 1:3]
    x = interior[:, 1:3]

    drift_x = x @ H.T
    drift_alpha = alpha.view(1, 2) @ M.T
    drift_total = drift_x + drift_alpha

    diffusion_term = 0.5 * torch.einsum("ij,bij->b", sigma_sq, hessian_x).unsqueeze(1)
    drift_term = torch.sum(u_x * drift_total, dim=1, keepdim=True)

    x_cost = torch.einsum("bi,ij,bj->b", x, C, x).unsqueeze(1)
    alpha_cost = torch.einsum("i,ij,j->", alpha, D, alpha).view(1, 1).expand_as(x_cost)

    residual = u_t + diffusion_term + drift_term + x_cost + alpha_cost
    eqn_loss = torch.mean(residual ** 2)

    u_T = model(terminal)
    x_T = terminal[:, 1:3]
    terminal_target = torch.einsum("bi,ij,bj->b", x_T, R, x_T).unsqueeze(1)
    boundary_loss = torch.mean((u_T - terminal_target) ** 2)

    total_loss = eqn_loss + boundary_loss
    return total_loss, eqn_loss, boundary_loss


# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    # Problem setup
    H_np = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
    M_np = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sigma_np = np.array([[0.2, 0.0], [0.0, 0.2]], dtype=np.float64)
    C_np = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    D_np = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    R_np = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    alpha_np = np.array([1.0, 1.0], dtype=np.float64)
    T_val = 1.0
    x0 = np.array([1.0, -1.0], dtype=np.float64)

    H = torch.tensor(H_np, dtype=torch.float32, device=device)
    M = torch.tensor(M_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    C = torch.tensor(C_np, dtype=torch.float32, device=device)
    D = torch.tensor(D_np, dtype=torch.float32, device=device)
    R = torch.tensor(R_np, dtype=torch.float32, device=device)
    alpha = torch.tensor(alpha_np, dtype=torch.float32, device=device)

    print("Computing Monte Carlo benchmark for constant control alpha = (1,1)^T ...")
    mc_benchmark = run_mc_constant_alpha(
        H=H_np,
        M=M_np,
        sigma=sigma_np,
        C=C_np,
        D=D_np,
        R=R_np,
        alpha=alpha_np,
        x0=x0,
        T=T_val,
        N_steps=5000,
        N_samples=100000,
        device=device,
    )
    print(f"Monte Carlo benchmark u(0, x0) = {mc_benchmark:.6f}")

    model = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    epochs = 4000
    batch_interior = 2048
    batch_terminal = 2048
    eval_every = 100

    total_loss_hist = []
    eqn_loss_hist = []
    boundary_loss_hist = []

    eval_epochs = []
    rel_error_hist = []

    print(f"Starting DGM training for {epochs} epochs...")

    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss, eqn_loss, boundary_loss = dgm_losses(
            model=model,
            H=H,
            M=M,
            sigma=sigma,
            C=C,
            D=D,
            R=R,
            alpha=alpha,
            T=T_val,
            batch_interior=batch_interior,
            batch_terminal=batch_terminal,
            device=device,
        )

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss_hist.append(total_loss.item())
        eqn_loss_hist.append(eqn_loss.item())
        boundary_loss_hist.append(boundary_loss.item())

        if (epoch + 1) % eval_every == 0:
            with torch.no_grad():
                inp_ref = torch.tensor([[0.0, x0[0], x0[1]]], dtype=torch.float32, device=device)
                u_pred = model(inp_ref).item()
                rel_err = abs(u_pred - mc_benchmark) / abs(mc_benchmark)

            eval_epochs.append(epoch + 1)
            rel_error_hist.append(rel_err)

        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch [{epoch + 1:4d}/{epochs}] | "
                f"Total Loss: {total_loss.item():.6e} | "
                f"Eqn Loss: {eqn_loss.item():.6e} | "
                f"Boundary Loss: {boundary_loss.item():.6e} | "
                f"Rel Error: {rel_error_hist[-1]:.6e}"
            )

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(total_loss_hist, label="Total DGM Loss")
    plt.yscale("log")
    plt.title("DGM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, rel_error_hist, marker="o", markersize=3, label="Relative Error vs MC")
    plt.yscale("log")
    plt.title("DGM Relative Error vs Monte Carlo")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()

    save_path = os.path.join("plots", "ex3_dgm_results.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved to '{save_path}'.")

    print("\nSuggested table entries every 500 epochs:")
    print("Epoch | Total Loss | Eqn Residual | Boundary Residual | Relative Error")
    for k in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
        idx = k - 1
        eval_idx = eval_epochs.index(k)
        print(
            f"{k:5d} | "
            f"{total_loss_hist[idx]:.6e} | "
            f"{eqn_loss_hist[idx]:.6e} | "
            f"{boundary_loss_hist[idx]:.6e} | "
            f"{rel_error_hist[eval_idx]:.6e}"
        )