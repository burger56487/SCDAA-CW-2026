import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 1. Define DGM Network Architecture
# ==========================================
class NetDGM(nn.Module):
    """
    Neural network to approximate the PDE solution u(t,x).
    Uses 3 hidden layers with 100 neurons each to ensure sufficient 
    capacity for learning second-order derivatives.
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

    def forward(self, t, x):
        tx = torch.cat([t, x], dim=1)
        return self.net(tx)

# ==========================================
# 2. Monte Carlo Benchmark Solution
# ==========================================
def run_mc_constant_alpha(x0, T, N_steps, N_samples, H_np, M_np, sigma_np, C_np, D_np, R_np):
    """
    Runs a Monte Carlo simulation with a constant control policy alpha=[1, 1]^T
    to serve as the ground truth benchmark for evaluating DGM.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = T / N_steps
    
    H = torch.tensor(H_np, dtype=torch.float32, device=device)
    M = torch.tensor(M_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    C = torch.tensor(C_np, dtype=torch.float32, device=device)
    D = torch.tensor(D_np, dtype=torch.float32, device=device)
    R_mat = torch.tensor(R_np, dtype=torch.float32, device=device)

    X = torch.tensor(x0, dtype=torch.float32, device=device).repeat(N_samples, 1).unsqueeze(-1)
    total_cost = torch.zeros(N_samples, 1, 1, device=device)
    
    # [cite_start]Constant control policy alpha = [1, 1]^T [cite: 91, 93, 102]
    a_col = torch.ones(N_samples, 2, 1, dtype=torch.float32, device=device)
    
    for _ in range(N_steps):
        cost_X = torch.bmm(torch.bmm(X.transpose(1, 2), C.expand(N_samples, 2, 2)), X)
        cost_a = torch.bmm(torch.bmm(a_col.transpose(1, 2), D.expand(N_samples, 2, 2)), a_col)
        total_cost += (cost_X + cost_a) * dt
        
        dW = torch.randn(N_samples, 2, 1, device=device) * np.sqrt(dt)
        drift = torch.bmm(H.expand(N_samples, 2, 2), X) + torch.bmm(M.expand(N_samples, 2, 2), a_col)
        diffusion = torch.bmm(sigma.expand(N_samples, 2, 2), dW)
        X = X + drift * dt + diffusion
        
    term_cost = torch.bmm(torch.bmm(X.transpose(1, 2), R_mat.expand(N_samples, 2, 2)), X)
    total_cost += term_cost
    return total_cost.mean().item()

# ==========================================
# 3. Deep Galerkin Method (DGM) Loss Function
# ==========================================
def dgm_loss(model, t, x, H, M, sigma, C, D, R_mat, T):
    batch_size = t.shape[0]
    
    # Gradients must be enabled for autograd to compute PDE derivatives
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    u = model(t, x) # Shape: (Batch, 1)
    
    # ---------------- 1. Exact Computation of 1st and 2nd Derivatives ----------------
    du = torch.autograd.grad(u, inputs=[t, x], grad_outputs=torch.ones_like(u), create_graph=True)
    u_t = du[0] # Shape: (Batch, 1)
    u_x = du[1] # Shape: (Batch, 2)
    
    # [Crucial Fix]: Extract each dimension of the first derivative before deriving 
    # to prevent incorrect Hessian computation due to cross-batch summation.
    du_dx1 = u_x[:, 0:1]
    du_dx2 = u_x[:, 1:2]
    d2u_dx1dx = torch.autograd.grad(du_dx1, x, grad_outputs=torch.ones_like(du_dx1), create_graph=True)[0]
    d2u_dx2dx = torch.autograd.grad(du_dx2, x, grad_outputs=torch.ones_like(du_dx2), create_graph=True)[0]
    
    d2u_dx1x1 = d2u_dx1dx[:, 0:1]
    d2u_dx1x2 = d2u_dx1dx[:, 1:2]
    d2u_dx2x1 = d2u_dx2dx[:, 0:1]
    d2u_dx2x2 = d2u_dx2dx[:, 1:2]

    sigma_sq = sigma @ sigma.T
    trace_term = 0.5 * (
        sigma_sq[0, 0] * d2u_dx1x1 + sigma_sq[0, 1] * d2u_dx2x1 +
        sigma_sq[1, 0] * d2u_dx1x2 + sigma_sq[1, 1] * d2u_dx2x2
    )

    # ---------------- 2. Elegant BMM Matrix Operations ----------------
    u_x_row = u_x.unsqueeze(1) # (B, 1, 2)
    x_col = x.unsqueeze(-1)    # (B, 2, 1)
    
    # (\partial_x u)^T H x
    term_Hx = torch.bmm(torch.bmm(u_x_row, H.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    
    # (\partial_x u)^T M \alpha (where \alpha = [1, 1]^T)
    alpha = torch.ones(batch_size, 2, 1, device=t.device)
    term_Malpha = torch.bmm(torch.bmm(u_x_row, M.expand(batch_size, 2, 2)), alpha).squeeze(-1)
    
    # x^T C x
    x_row = x.unsqueeze(1)
    term_Cx = torch.bmm(torch.bmm(x_row, C.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    
    # \alpha^T D \alpha
    alpha_row = alpha.transpose(1, 2)
    term_Dalpha = torch.bmm(torch.bmm(alpha_row, D.expand(batch_size, 2, 2)), alpha).squeeze(-1)
    
    # [cite_start]PDE Interior Equation Residual [cite: 92, 95]
    residual = u_t + trace_term + term_Hx + term_Malpha + term_Cx + term_Dalpha
    loss_eqn = torch.mean(residual**2)
    
    # ---------------- 3. Terminal Boundary Condition Residual ----------------
    t_T = torch.full_like(t, T, requires_grad=False)
    x_T = (torch.rand(batch_size, 2, device=t.device) * 6.0) - 3.0 # Independent sampling at terminal boundary
    
    u_T_pred = model(t_T, x_T)
    
    x_T_row = x_T.unsqueeze(1)
    x_T_col = x_T.unsqueeze(-1)
    u_T_true = torch.bmm(torch.bmm(x_T_row, R_mat.expand(batch_size, 2, 2)), x_T_col).squeeze(-1)
    
    loss_bound = torch.mean((u_T_pred - u_T_true)**2)
    
    return loss_eqn + loss_bound, loss_eqn.item(), loss_bound.item()

# ==========================================
# 4. Main Program: DGM Training and MC Validation
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
    test_x0 = [1.0, -1.0]

    # Convert numpy arrays to PyTorch Tensors
    H_t = torch.tensor(H_mat, dtype=torch.float32, device=device)
    M_t = torch.tensor(M_mat, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma_mat, dtype=torch.float32, device=device)
    C_t = torch.tensor(C_mat, dtype=torch.float32, device=device)
    D_t = torch.tensor(D_mat, dtype=torch.float32, device=device)
    R_t = torch.tensor(R_mat, dtype=torch.float32, device=device)

    # Precompute MC ground truth for comparison
    print("Computing MC benchmark value (Constant policy alpha=[1,1]^T)...")
    mc_benchmark = run_mc_constant_alpha(
        test_x0, T_val, N_steps=1000, N_samples=100000, 
        H_np=H_mat, M_np=M_mat, sigma_np=sigma_mat, 
        C_np=C_mat, D_np=D_mat, R_np=R_mat
    )
    print(f"MC Ground Truth u(0, x0): {mc_benchmark:.6f}")

    # Initialize Network, Optimizer, and Learning Rate Scheduler
    model = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)

    epochs = 4000
    batch_size = 2048
    history_loss = []
    history_mc_error = []
    eval_epochs = []

    print(f"\nStarting DGM Training ({epochs} epochs in total)...")
    
    for epoch in range(epochs):
        # Explicitly set the model to training mode
        model.train() 
        optimizer.zero_grad()
        
        # Random sampling within the spatial and temporal domain
        t_batch = torch.rand((batch_size, 1), dtype=torch.float32, device=device) * T_val
        x_batch = torch.rand((batch_size, 2), dtype=torch.float32, device=device) * 6.0 - 3.0
        
        loss, l_eqn, l_bound = dgm_loss(model, t_batch, x_batch, H_t, M_t, sigma_t, C_t, D_t, R_t, T_val)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        history_loss.append(loss.item())
        
        # Periodically evaluate relative error against the MC benchmark
        if (epoch + 1) % 100 == 0:
            # Switch to evaluation mode
            model.eval() 
            with torch.no_grad():
                test_t = torch.tensor([[0.0]], dtype=torch.float32, device=device)
                test_x = torch.tensor([test_x0], dtype=torch.float32, device=device)
                u_pred = model(test_t, test_x).item()
                rel_error = abs(u_pred - mc_benchmark) / abs(mc_benchmark)
                history_mc_error.append(rel_error)
                eval_epochs.append(epoch + 1)
            
            if (epoch + 1) % 500 == 0:
                print(f"Epoch [{epoch+1:4d}/{epochs}] | Loss: {loss.item():.4e} (Eqn: {l_eqn:.4e}, Bd: {l_bound:.4e}) | Rel Error: {rel_error:.4%}")

    # ---------------- 5. Plot and Save Results ----------------
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_loss, color='purple', alpha=0.8, label='Total DGM Loss')
    plt.yscale('log')
    plt.title('DGM Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, history_mc_error, 'o-', color='teal', label='Relative Error vs MC')
    plt.yscale('log')
    plt.title('DGM Relative Error Evaluated against Monte Carlo')
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.tight_layout()
    plt.savefig('ex3_dgm_results.png', dpi=300)
    print("\n✅ DGM training complete! The loss and error comparison plots have been saved as 'ex3_dgm_results.png'.")