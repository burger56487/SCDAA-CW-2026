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
# 1. Networks
# ==========================================
class NetDGM(nn.Module):
    """
    Value network for policy evaluation.
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


class NetFFN(nn.Module):
    """
    Policy network for Markov control.
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


def initialise_constant_policy(policy_net, constant=(1.0, 1.0)):
    """
    Initialise the policy network near a constant policy a ≈ (1,1),
    but keep nonzero weights so that the network remains trainable
    as a function of (t, x).
    """
    last_linear = None

    for module in policy_net.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
            last_linear = module

    if last_linear is not None:
        last_linear.bias.data = torch.tensor(
            constant, dtype=torch.float32, device=last_linear.bias.device
        )

# ==========================================
# 2. Sampling / Derivatives
# ==========================================
def sample_interior(batch_size, T, device):
    t = torch.rand(batch_size, 1, dtype=torch.float32, device=device) * T
    x = torch.rand(batch_size, 2, dtype=torch.float32, device=device) * 6.0 - 3.0
    inputs = torch.cat([t, x], dim=1)
    inputs.requires_grad_(True)
    return inputs


def sample_terminal(batch_size, T, device):
    t = torch.full((batch_size, 1), T, dtype=torch.float32, device=device)
    x = torch.rand(batch_size, 2, dtype=torch.float32, device=device) * 6.0 - 3.0
    inputs = torch.cat([t, x], dim=1)
    return inputs


def compute_hessian_wrt_x(u, inputs):
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
# 3. Loss Functions
# ==========================================
def value_pde_loss(value_net, policy_net, H, M, sigma, C, D, R, T, batch_interior, batch_terminal, device):
    """
    Policy evaluation:
    Given current policy a_phi(t,x), solve the linear PDE using DGM.
    """
    sigma_sq = sigma @ sigma.T

    interior = sample_interior(batch_interior, T, device)
    terminal = sample_terminal(batch_terminal, T, device)

    u = value_net(interior)
    grad_all, hessian_x = compute_hessian_wrt_x(u, interior)

    u_t = grad_all[:, 0:1]
    u_x = grad_all[:, 1:3]
    x = interior[:, 1:3]

    with torch.no_grad():
        a = policy_net(interior.detach())

    drift_x = x @ H.T
    drift_a = a @ M.T
    drift_total = drift_x + drift_a

    diffusion_term = 0.5 * torch.einsum("ij,bij->b", sigma_sq, hessian_x).unsqueeze(1)
    drift_term = torch.sum(u_x * drift_total, dim=1, keepdim=True)

    x_cost = torch.einsum("bi,ij,bj->b", x, C, x).unsqueeze(1)
    a_cost = torch.einsum("bi,ij,bj->b", a, D, a).unsqueeze(1)

    residual = u_t + diffusion_term + drift_term + x_cost + a_cost
    eqn_loss = torch.mean(residual ** 2)

    u_T = value_net(terminal)
    x_T = terminal[:, 1:3]
    target_T = torch.einsum("bi,ij,bj->b", x_T, R, x_T).unsqueeze(1)
    boundary_loss = torch.mean((u_T - target_T) ** 2)

    total_loss = eqn_loss + boundary_loss
    return total_loss, eqn_loss, boundary_loss


def hamiltonian_loss(value_net, policy_net, H, M, C, D, T, batch_size, device):
    """
    Policy improvement:
    Minimize the empirical Hamiltonian with value_net fixed.
    """
    inputs = sample_interior(batch_size, T, device)
    x = inputs[:, 1:3]

    v = value_net(inputs)
    grad_all = torch.autograd.grad(
        v.sum(), inputs, create_graph=False, retain_graph=False
    )[0]
    v_x = grad_all[:, 1:3].detach()

    a = policy_net(inputs.detach())

    term1 = torch.sum(v_x * (x @ H.T), dim=1, keepdim=True)
    term2 = torch.sum(v_x * (a @ M.T), dim=1, keepdim=True)
    term3 = torch.einsum("bi,ij,bj->b", x, C, x).unsqueeze(1)
    term4 = torch.einsum("bi,ij,bj->b", a, D, a).unsqueeze(1)

    ham = term1 + term2 + term3 + term4
    return torch.mean(ham)


# ==========================================
# 4. Benchmark Comparison
# ==========================================
@torch.no_grad()
def evaluate_against_benchmark(value_net, policy_net, lqr_solver, x0, device):
    t_ref = torch.tensor([0.0], dtype=torch.float32, device=device)
    x_ref_batch = torch.tensor(np.array([[x0]]), dtype=torch.float32, device=device)  # (1,1,2)
    inp_ref = torch.tensor([[0.0, x0[0], x0[1]]], dtype=torch.float32, device=device)

    v_true = lqr_solver.value_function(t_ref, x_ref_batch).item()
    a_true = lqr_solver.markov_control(t_ref, x_ref_batch).squeeze(0).cpu().numpy()

    v_pred = value_net(inp_ref).item()
    a_pred = policy_net(inp_ref).squeeze(0).cpu().numpy()

    rel_err_v = abs(v_pred - v_true) / abs(v_true)
    rel_err_a = np.linalg.norm(a_pred - a_true) / np.linalg.norm(a_true)

    return v_true, a_true, v_pred, a_pred, rel_err_v, rel_err_a


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
    T_val = 1.0
    x0 = np.array([1.0, -1.0], dtype=np.float64)

    H = torch.tensor(H_np, dtype=torch.float32, device=device)
    M = torch.tensor(M_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    C = torch.tensor(C_np, dtype=torch.float32, device=device)
    D = torch.tensor(D_np, dtype=torch.float32, device=device)
    R = torch.tensor(R_np, dtype=torch.float32, device=device)

    # Riccati benchmark from Exercise 1.1
    print("Initializing Riccati benchmark...")
    lqr_solver = LQR_Solver(H_np, M_np, sigma_np, C_np, D_np, R_np, T_val)
    time_grid = np.linspace(0.0, T_val, 500)
    lqr_solver.solve_riccati(time_grid)
    print("Benchmark ready.")

    value_net = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    policy_net = NetFFN(input_dim=3, hidden_dim=100, output_dim=2).to(device)

    initialise_constant_policy(policy_net, constant=(1.0, 1.0))

    outer_steps = 10
    value_epochs_per_step = 500
    policy_epochs_per_step = 600

    batch_interior = 1024
    batch_terminal = 1024
    batch_policy = 2048

    value_predictions = []
    value_rel_errors = []
    action_predictions = []
    action_rel_errors = []

    for step in range(outer_steps):
        print(f"\n==============================")
        print(f"Policy Iteration Step {step + 1}/{outer_steps}")
        print(f"==============================")

        # -------------------------
        # Policy evaluation
        # -------------------------
        optimizer_v = optim.Adam(value_net.parameters(), lr=1e-3)

        value_net.train()
        policy_net.eval()

        for _ in range(value_epochs_per_step):
            optimizer_v.zero_grad()
            total_loss, eqn_loss, boundary_loss = value_pde_loss(
                value_net=value_net,
                policy_net=policy_net,
                H=H,
                M=M,
                sigma=sigma,
                C=C,
                D=D,
                R=R,
                T=T_val,
                batch_interior=batch_interior,
                batch_terminal=batch_terminal,
                device=device,
            )
            total_loss.backward()
            optimizer_v.step()

        print(
            f"Policy evaluation complete | "
            f"Last Total Loss: {total_loss.item():.6e} | "
            f"Eqn: {eqn_loss.item():.6e} | "
            f"Boundary: {boundary_loss.item():.6e}"
        )

        # -------------------------
        # Policy improvement
        # -------------------------
        optimizer_a = optim.Adam(policy_net.parameters(), lr=1e-3)

        value_net.eval()
        policy_net.train()

        for _ in range(policy_epochs_per_step):
            optimizer_a.zero_grad()
            ham_loss = hamiltonian_loss(
                value_net=value_net,
                policy_net=policy_net,
                H=H,
                M=M,
                C=C,
                D=D,
                T=T_val,
                batch_size=batch_policy,
                device=device,
            )
            ham_loss.backward()
            optimizer_a.step()

        print(f"Policy improvement complete | Last Hamiltonian Loss: {ham_loss.item():.6e}")

        # -------------------------
        # Benchmark comparison
        # -------------------------
        value_net.eval()
        policy_net.eval()

        v_true, a_true, v_pred, a_pred, rel_err_v, rel_err_a = evaluate_against_benchmark(
            value_net=value_net,
            policy_net=policy_net,
            lqr_solver=lqr_solver,
            x0=x0,
            device=device,
        )

        value_predictions.append(v_pred)
        value_rel_errors.append(rel_err_v)
        action_predictions.append(a_pred.copy())
        action_rel_errors.append(rel_err_a)

        print("Comparison against Riccati benchmark at (t, x) = (0, x0):")
        print(f"  True value      : {v_true:.6f}")
        print(f"  Predicted value : {v_pred:.6f}")
        print(f"  Relative error v: {rel_err_v:.6e}")
        print(f"  True action     : {a_true}")
        print(f"  Predicted action: {a_pred}")
        print(f"  Relative error a: {rel_err_a:.6e}")

    os.makedirs("plots", exist_ok=True)

    pi_steps = np.arange(1, outer_steps + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(pi_steps, value_rel_errors, marker="o", label="Value Relative Error")
    plt.yscale("log")
    plt.title("Policy Iteration: Value Function Error")
    plt.xlabel("PI Step")
    plt.ylabel("Relative Error (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.subplot(1, 2, 2)
    plt.plot(pi_steps, action_rel_errors, marker="o", label="Action Relative Error")
    plt.yscale("log")
    plt.title("Policy Iteration: Markov Control Error")
    plt.xlabel("PI Step")
    plt.ylabel("Relative Error (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()

    save_path = os.path.join("plots", "ex4_policy_iteration.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved to '{save_path}'.")

    print("\nSuggested table entries:")
    print("PI step | Pred. v(0,x0) | Rel. error v | Pred. a(0,x0) | Rel. error a")
    for i in range(outer_steps):
        a_pred = action_predictions[i]
        print(
            f"{i + 1:7d} | "
            f"{value_predictions[i]:13.6f} | "
            f"{value_rel_errors[i]:.6e} | "
            f"[{a_pred[0]:.6f}, {a_pred[1]:.6f}] | "
            f"{action_rel_errors[i]:.6e}"
        )