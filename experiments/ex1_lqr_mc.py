import numpy as np
import torch
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ==========================================
# 1. LQR Solver Class (Exercise 1.1)
# ==========================================
class LQR_Solver:
    def __init__(self, H, M, sigma, C, D, R, T):
        self.H = np.array(H, dtype=np.float32)
        self.M = np.array(M, dtype=np.float32)
        self.sigma = np.array(sigma, dtype=np.float32)
        self.C = np.array(C, dtype=np.float32)
        self.D = np.array(D, dtype=np.float32)
        self.R = np.array(R, dtype=np.float32)
        self.T = T
        self.D_inv = np.linalg.inv(self.D)
        self.sigma_sq = self.sigma @ self.sigma.T
        self.S_interp = None
        self.integral_interp = None

    def riccati_ode(self, t, S_flat):
        S = S_flat.reshape((2, 2))
        dS = -2 * self.H @ S + S @ self.M @ self.D_inv @ self.M.T @ S - self.C
        return dS.flatten()

    def solve_riccati(self, time_grid):
        time_grid_rev = np.flip(time_grid)
        S_T_flat = self.R.flatten()
        
        # Optimization 1: Increase Riccati ODE solving precision to prevent MC error 
        # convergence curve from flattening prematurely.
        sol = solve_ivp(
            fun=self.riccati_ode,
            t_span=(self.T, 0.0),
            y0=S_T_flat,
            t_eval=time_grid_rev,
            method='RK45',
            rtol=1e-8,  # Increased relative tolerance
            atol=1e-8   # Increased absolute tolerance
        )
        t_forward = np.flip(sol.t)
        S_forward = np.flip(sol.y, axis=1)
        self.S_interp = interp1d(t_forward, S_forward, kind='cubic', axis=1, fill_value="extrapolate")
        
        tr_vals = np.zeros_like(t_forward)
        for i, t in enumerate(t_forward):
            S_t = S_forward[:, i].reshape((2, 2))
            tr_vals[i] = np.trace(self.sigma_sq @ S_t)
            
        cum_int = cumulative_trapezoid(tr_vals, t_forward, initial=0.0)
        integral_vals = cum_int[-1] - cum_int
        self.integral_interp = interp1d(t_forward, integral_vals, kind='cubic', fill_value="extrapolate")

    def _get_S_and_integral(self, t_batch):
        t_np = t_batch.detach().cpu().numpy()
        S_flat_np = self.S_interp(t_np)
        integral_np = self.integral_interp(t_np)
        S_tensor = torch.tensor(S_flat_np.T, dtype=torch.float32, device=t_batch.device).reshape(-1, 2, 2)
        integral_tensor = torch.tensor(integral_np, dtype=torch.float32, device=t_batch.device).reshape(-1, 1)
        return S_tensor, integral_tensor

    def value_function(self, t_batch, x_batch):
        S_t, integral_t = self._get_S_and_integral(t_batch)
        x_S_x = torch.bmm(torch.bmm(x_batch, S_t), x_batch.mT)
        v_val = x_S_x.squeeze(2) + integral_t
        return v_val
        
    def markov_control(self, t_batch, x_batch):
        """
        Required for Exercise 1.1 / 2.2: Returns optimal Markov control a(t,x)
        """
        S_t, _ = self._get_S_and_integral(t_batch)
        K_np = -self.D_inv @ self.M.T
        K = torch.tensor(K_np, dtype=torch.float32, device=t_batch.device).unsqueeze(0)
        x_col = x_batch.mT 
        S_x = torch.bmm(S_t, x_col)
        a_val = torch.matmul(K, S_x) 
        return a_val.squeeze(-1)

# ==========================================
# 2. Monte Carlo Simulation Core (Exercise 1.2)
# ==========================================
def run_mc_lqr(lqr_solver, x0, T, N_steps, N_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = T / N_steps
    sqrt_dt = np.sqrt(dt)

    # Optimization 2: Move .expand() out of the loop to avoid reallocating view objects in every iteration
    H_batch = torch.tensor(lqr_solver.H, dtype=torch.float32, device=device).expand(N_samples, 2, 2)
    M_batch = torch.tensor(lqr_solver.M, dtype=torch.float32, device=device).expand(N_samples, 2, 2)
    sigma_batch = torch.tensor(lqr_solver.sigma, dtype=torch.float32, device=device).expand(N_samples, 2, 2)
    C_batch = torch.tensor(lqr_solver.C, dtype=torch.float32, device=device).expand(N_samples, 2, 2)
    D_batch = torch.tensor(lqr_solver.D, dtype=torch.float32, device=device).expand(N_samples, 2, 2)
    R_batch = torch.tensor(lqr_solver.R, dtype=torch.float32, device=device).expand(N_samples, 2, 2)

    # Optimization 3: Precompute S(t) and control gain K_S for the entire time grid at once, eliminating CPU-GPU communication inside the loop
    time_grid = np.linspace(0, T, N_steps, endpoint=False)
    S_flat_np = lqr_solver.S_interp(time_grid) # Call SciPy only once
    S_seq = torch.tensor(S_flat_np.T, dtype=torch.float32, device=device).reshape(N_steps, 2, 2)
    
    K_np = -lqr_solver.D_inv @ lqr_solver.M.T
    K_tensor = torch.tensor(K_np, dtype=torch.float32, device=device)
    
    # Control law a = K @ S(t) @ X, precompute KS(t) = K @ S(t)
    KS_seq = torch.matmul(K_tensor, S_seq)

    X = torch.tensor(x0, dtype=torch.float32, device=device).repeat(N_samples, 1).unsqueeze(-1)
    total_cost = torch.zeros(N_samples, 1, 1, device=device)
    
    for i in range(N_steps):
        # Extract the control gain for the current step directly from the precomputed GPU tensor by index
        KS_i = KS_seq[i].expand(N_samples, 2, 2)
        
        # Calculate control vector a
        a_col = torch.bmm(KS_i, X)
        
        X_T = X.transpose(1, 2)
        a_T = a_col.transpose(1, 2)
        
        # Running cost
        cost_X = torch.bmm(torch.bmm(X_T, C_batch), X)
        cost_a = torch.bmm(torch.bmm(a_T, D_batch), a_col)
        total_cost += (cost_X + cost_a) * dt
        
        # Euler-Maruyama step (Explicit)
        dW = torch.randn(N_samples, 2, 1, device=device) * sqrt_dt
        drift = torch.bmm(H_batch, X) + torch.bmm(M_batch, a_col)
        diffusion = torch.bmm(sigma_batch, dW)
        X = X + drift * dt + diffusion
        
    term_cost = torch.bmm(torch.bmm(X.transpose(1, 2), R_batch), X)
    total_cost += term_cost
    return total_cost.mean().item()

# ==========================================
# 3. Plotting and Testing Function (Exercise 1.2)
# ==========================================
def plot_convergence(lqr_solver, x0, T):
    t_0 = torch.tensor([0.0], dtype=torch.float32)
    x_0_tensor = torch.tensor([[x0]], dtype=torch.float32)
    v_true = lqr_solver.value_function(t_0, x_0_tensor).item()
    print(f"Theoretical true value v(0, x_0): {v_true:.6f}")

    N_samples_fixed = 10**5
    # ADDED MORE DATA POINTS for higher precision rendering
    N_steps_list = [1, 2, 4, 8, 15, 30, 60, 120, 250, 500, 1000, 2000, 3000, 4000, 5000] 
    errors_time = []
    
    print("\n--- Starting Test: Time Discretization Convergence ---")
    for n_step in N_steps_list:
        v_mc = run_mc_lqr(lqr_solver, x0, T, N_steps=n_step, N_samples=N_samples_fixed)
        err = abs(v_mc - v_true)
        errors_time.append(err)
        print(f"N_steps: {n_step:4d} | MC Value: {v_mc:.4f} | Absolute Error: {err:.6f}")

    N_steps_fixed = 5000
    # ADDED MORE DATA POINTS for higher precision rendering
    N_samples_list = [10, 20, 40, 80, 150, 300, 600, 1200, 2500, 5000, 10000, 20000, 40000, 60000, 80000, 100000]
    errors_samples = []

    print("\n--- Starting Test: Monte Carlo Sample Size Convergence ---")
    for n_samp in N_samples_list:
        v_mc = run_mc_lqr(lqr_solver, x0, T, N_steps=N_steps_fixed, N_samples=n_samp)
        err = abs(v_mc - v_true)
        errors_samples.append(err)
        print(f"N_samples: {n_samp:6d} | MC Value: {v_mc:.4f} | Absolute Error: {err:.6f}")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.loglog(N_steps_list, errors_time, 'o-', label='MC Error')
    # Use index 2 (N_steps=10) to anchor the reference line
    ref_line_1 = [errors_time[2] * (N_steps_list[2] / n) for n in N_steps_list]
    plt.loglog(N_steps_list, ref_line_1, 'r--', label='Reference O(1/N_steps)')
    plt.title('Convergence of Time Discretization')
    plt.xlabel('Number of Time Steps')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.subplot(1, 2, 2)
    plt.loglog(N_samples_list, errors_samples, 's-', color='orange', label='MC Error')
    # Use index 3 (N_samples=100) to anchor the reference line
    ref_line_2 = [errors_samples[3] * np.sqrt(N_samples_list[3] / n) for n in N_samples_list]
    plt.loglog(N_samples_list, ref_line_2, 'r--', label='Reference O(1/sqrt(N_samples))')
    plt.title('Convergence of Monte Carlo Sampling')
    plt.xlabel('Number of MC Samples')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('convergence_plot.png', dpi=300)
    print("\n✅ The plot has been successfully saved as 'convergence_plot.png' in the current directory.")

# ==========================================
# 4. Main Program Entry
# ==========================================
if __name__ == "__main__":
    H_mat = np.array([[0.1, 0.0], [0.0, 0.1]])
    M_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma_mat = np.array([[0.2, 0.0], [0.0, 0.2]])
    C_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    D_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    T_val = 1.0

    print("Initializing LQR solver and solving Riccati ODE...")
    lqr_solver = LQR_Solver(H_mat, M_mat, sigma_mat, C_mat, D_mat, R_mat, T_val)
    time_grid = np.linspace(0, T_val, 500)
    lqr_solver.solve_riccati(time_grid)
    print("Riccati ODE solved successfully!")

    x0_val = [1.0, -1.0] 
    print("Starting Monte Carlo validation. This requires some computational power, please wait...")
    plot_convergence(lqr_solver, x0_val, T_val)