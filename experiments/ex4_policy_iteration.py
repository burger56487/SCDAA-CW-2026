import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import your implemented LQR_Solver as the final testing benchmark
from ex1_lqr_mc import LQR_Solver

# ==========================================
# 1. Neural Network Architectures
# ==========================================
class NetDGM(nn.Module):
    """
    Value network v(t,x): 3 hidden layers to ensure sufficient capacity 
    to fit the second-order derivatives (Hessian).
    """
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super(NetDGM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class NetFFN(nn.Module):
    """
    Policy network a(t,x): 2 hidden layers to approximate the Markov control.
    """
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super(NetFFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Loss Functions (Fully Decoupled)
# ==========================================
def pde_loss(model_v, model_a, t, x, H, M, sigma, C, D, R_mat, T):
    """
    Policy Evaluation: Fix policy 'a', solve for 'V' using DGM PDE Loss.
    """
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    inputs = torch.cat([t, x], dim=1)
    u = model_v(inputs)
    
    # 🌟 Core Fix: Correctly split and compute Hessian to avoid cross-batch summation bug
    du = torch.autograd.grad(u, inputs=[t, x], grad_outputs=torch.ones_like(u), create_graph=True)
    u_t, u_x = du[0], du[1]
    
    du_dx1 = u_x[:, 0:1]
    du_dx2 = u_x[:, 1:2]
    d2u_dx1dx = torch.autograd.grad(du_dx1, x, grad_outputs=torch.ones_like(du_dx1), create_graph=True)[0]
    d2u_dx2dx = torch.autograd.grad(du_dx2, x, grad_outputs=torch.ones_like(du_dx2), create_graph=True)[0]
    
    sigma_sq = sigma @ sigma.T
    trace_term = 0.5 * (
        sigma_sq[0, 0] * d2u_dx1dx[:, 0:1] + sigma_sq[0, 1] * d2u_dx2dx[:, 0:1] +
        sigma_sq[1, 0] * d2u_dx1dx[:, 1:2] + sigma_sq[1, 1] * d2u_dx2dx[:, 1:2]
    )
    
    # Get current policy network prediction (detach gradients, do not update a_net)
    with torch.no_grad():
        a_pred = model_a(inputs)
    a_col = a_pred.unsqueeze(-1)
            
    # Elegant Batch Matrix Multiplication (BMM)
    batch_size = t.shape[0]
    u_x_row = u_x.unsqueeze(1)
    x_col = x.unsqueeze(-1)
    x_row = x.unsqueeze(1)
    a_row = a_col.transpose(1, 2)
    
    term_Hx = torch.bmm(torch.bmm(u_x_row, H.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    term_Ma = torch.bmm(torch.bmm(u_x_row, M.expand(batch_size, 2, 2)), a_col).squeeze(-1)
    term_Cx = torch.bmm(torch.bmm(x_row, C.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    term_Da = torch.bmm(torch.bmm(a_row, D.expand(batch_size, 2, 2)), a_col).squeeze(-1)
    
    # Assemble interior residual
    residual = u_t + trace_term + term_Hx + term_Ma + term_Cx + term_Da
    loss_eqn = torch.mean(residual**2)
    
    # Assemble boundary conditions
    t_T = torch.full_like(t, T, requires_grad=False)
    x_T = (torch.rand(batch_size, 2, device=t.device) * 6.0) - 3.0 # Independent sampling at boundary
    u_T_pred = model_v(torch.cat([t_T, x_T], dim=1))
    
    x_T_row = x_T.unsqueeze(1)
    x_T_col = x_T.unsqueeze(-1)
    u_T_true = torch.bmm(torch.bmm(x_T_row, R_mat.expand(batch_size, 2, 2)), x_T_col).squeeze(-1)
    loss_bound = torch.mean((u_T_pred - u_T_true)**2)
    
    return loss_eqn + loss_bound

def hamiltonian_loss(model_v, model_a, t, x, H, M, C, D):
    """
    Policy Improvement: Fix V, minimize Hamiltonian to update policy 'a'.
    """
    t.requires_grad_(True)
    x.requires_grad_(True)
    inputs = torch.cat([t, x], dim=1)
    
    # 1. Compute \partial_x v (Fix v, detach its computation graph)
    u = model_v(inputs)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=False)[0]
    u_x = u_x.detach() # 🌟 Detach the value function's gradient graph
    
    # 2. Compute current policy network output (enable gradients for optimizer_a)
    a_pred = model_a(inputs)
    a_col = a_pred.unsqueeze(-1)
    
    batch_size = t.shape[0]
    u_x_row = u_x.unsqueeze(1)
    x_col = x.unsqueeze(-1)
    x_row = x.unsqueeze(1)
    a_row = a_col.transpose(1, 2)
    
    term_Hx = torch.bmm(torch.bmm(u_x_row, H.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    term_Ma = torch.bmm(torch.bmm(u_x_row, M.expand(batch_size, 2, 2)), a_col).squeeze(-1)
    term_Cx = torch.bmm(torch.bmm(x_row, C.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    term_Da = torch.bmm(torch.bmm(a_row, D.expand(batch_size, 2, 2)), a_col).squeeze(-1)
    
    # 🌟 Hamiltonian (No need for MSE, directly minimize its expected value)
    hamiltonian = term_Hx + term_Ma + term_Cx + term_Da
    return torch.mean(hamiltonian)

# ==========================================
# 3. Main Program: Policy Iteration Training and Validation
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current computation device: {device}")

    # System Matrices
    H_mat = np.array([[0.1, 0.0], [0.0, 0.1]])
    M_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma_mat = np.array([[0.2, 0.0], [0.0, 0.2]])
    C_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    D_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    T_val = 1.0
    x0_val = [1.0, -1.0]

    # Convert to Tensors
    H_t = torch.tensor(H_mat, dtype=torch.float32, device=device)
    M_t = torch.tensor(M_mat, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma_mat, dtype=torch.float32, device=device)
    C_t = torch.tensor(C_mat, dtype=torch.float32, device=device)
    D_t = torch.tensor(D_mat, dtype=torch.float32, device=device)
    R_t = torch.tensor(R_mat, dtype=torch.float32, device=device)

    # ---------------- LQR Ground Truth Benchmark ----------------
    print("Computing theoretical optimal LQR solution as benchmark...")
    lqr_solver = LQR_Solver(H_mat, M_mat, sigma_mat, C_mat, D_mat, R_mat, T_val)
    lqr_solver.solve_riccati(np.linspace(0, T_val, 500))
    test_t = torch.tensor([0.0], dtype=torch.float32, device=device)
    test_x = torch.tensor([[x0_val]], dtype=torch.float32, device=device)
    
    true_v = lqr_solver.value_function(test_t, test_x).item()
    true_a = lqr_solver.markov_control(test_t, test_x).detach().cpu().numpy()[0]
    print(f"Theoretical Optimal v(0,x0): {true_v:.6f} | a(0,x0): {true_a}")
    print("========================================")

    # ---------------- Initialize Networks and Optimizers ----------------
    model_v = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    model_a = NetFFN(input_dim=3, hidden_dim=100, output_dim=2).to(device)
    
    opt_v = optim.Adam(model_v.parameters(), lr=1e-3)
    opt_a = optim.Adam(model_a.parameters(), lr=1e-3)
    
    # Add Learning Rate Schedulers
    scheduler_v = optim.lr_scheduler.StepLR(opt_v, step_size=3000, gamma=0.5)
    scheduler_a = optim.lr_scheduler.StepLR(opt_a, step_size=2000, gamma=0.5)

    # Training Hyperparameters
    pi_iterations = 10      # INCREASED to 10 for more data points and smoother plots
    epochs_pe = 1500        # Inner epochs for Policy Evaluation
    epochs_pi = 1000        # Inner epochs for Policy Improvement
    batch_size = 2048

    error_v_history = []
    error_a_history = []

    # ---------------- Start Policy Iteration ----------------
    for pi_step in range(pi_iterations):
        print(f"\n---> Starting Policy Iteration Step {pi_step+1}/{pi_iterations}")
        
        # 1. Policy Evaluation
        model_v.train()
        for _ in range(epochs_pe):
            opt_v.zero_grad()
            t_batch = torch.rand((batch_size, 1), dtype=torch.float32, device=device) * T_val
            x_batch = torch.rand((batch_size, 2), dtype=torch.float32, device=device) * 6.0 - 3.0
            loss_v = pde_loss(model_v, model_a, t_batch, x_batch, H_t, M_t, sigma_t, C_t, D_t, R_t, T_val)
            loss_v.backward()
            opt_v.step()
            scheduler_v.step()
            
        # 2. Policy Improvement
        model_a.train()
        for _ in range(epochs_pi):
            opt_a.zero_grad()
            t_batch = torch.rand((batch_size, 1), dtype=torch.float32, device=device) * T_val
            x_batch = torch.rand((batch_size, 2), dtype=torch.float32, device=device) * 6.0 - 3.0
            loss_ham = hamiltonian_loss(model_v, model_a, t_batch, x_batch, H_t, M_t, C_t, D_t)
            loss_ham.backward()
            opt_a.step()
            scheduler_a.step()

        # 3. Evaluate error against true LQR solution
        model_v.eval()
        model_a.eval()
        with torch.no_grad():
            t_eval = torch.tensor([[0.0]], dtype=torch.float32, device=device)
            x_eval = torch.tensor([x0_val], dtype=torch.float32, device=device)
            
            pred_v = model_v(torch.cat([t_eval, x_eval], dim=1)).item()
            pred_a = model_a(torch.cat([t_eval, x_eval], dim=1)).cpu().numpy()[0]
            
            # Calculate Relative Error
            err_v = abs(pred_v - true_v) / abs(true_v)
            err_a = np.linalg.norm(pred_a - true_a) / np.linalg.norm(true_a)
            
            error_v_history.append(err_v)
            error_a_history.append(err_a)
            
            print(f"Evaluation -> v_pred: {pred_v:.4f} (Rel. Error: {err_v:.2%}) | a_pred: {pred_a} (Rel. Error: {err_a:.2%})")

    # ---------------- Plot and Save Final Charts ----------------
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, pi_iterations+1), error_v_history, 'b-o', label='Relative Error of v(t,x)')
    plt.yscale('log')
    plt.title('Convergence of Value Function (NetDGM)')
    plt.xlabel('Policy Iteration Step')
    plt.ylabel('Relative Error vs LQR Optimal (Log Scale)')
    plt.xticks(range(1, pi_iterations + 1))
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, pi_iterations+1), error_a_history, 'r-s', label='Relative Error of a(t,x)')
    plt.yscale('log')
    plt.title('Convergence of Markov Control (NetFFN)')
    plt.xlabel('Policy Iteration Step')
    plt.ylabel('Relative Error vs LQR Optimal (Log Scale)')
    plt.xticks(range(1, pi_iterations + 1))
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.tight_layout()
    plt.savefig('ex4_policy_iteration.png', dpi=300)
    print("\n✅ PIA Training complete! The convergence plot has been saved as 'ex4_policy_iteration.png'.")