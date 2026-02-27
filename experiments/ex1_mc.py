import numpy as np
import torch
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ==========================================
# 1. LQR 求解器类 (Exercise 1.1)
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
        sol = solve_ivp(
            fun=self.riccati_ode,
            t_span=(self.T, 0.0),
            y0=S_T_flat,
            t_eval=time_grid_rev,
            method='RK45'
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
        S_t, _ = self._get_S_and_integral(t_batch)
        K_np = -self.D_inv @ self.M.T
        K = torch.tensor(K_np, dtype=torch.float32, device=t_batch.device).unsqueeze(0)
        x_col = x_batch.mT 
        S_x = torch.bmm(S_t, x_col)
        a_val = torch.matmul(K, S_x) 
        return a_val.squeeze(-1)

# ==========================================
# 2. 蒙特卡洛模拟核心 (Exercise 1.2)
# ==========================================
def run_mc_lqr(lqr_solver, x0, T, N_steps, N_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = T / N_steps
    H = torch.tensor(lqr_solver.H, dtype=torch.float32, device=device)
    M = torch.tensor(lqr_solver.M, dtype=torch.float32, device=device)
    sigma = torch.tensor(lqr_solver.sigma, dtype=torch.float32, device=device)
    C = torch.tensor(lqr_solver.C, dtype=torch.float32, device=device)
    D = torch.tensor(lqr_solver.D, dtype=torch.float32, device=device)
    R = torch.tensor(lqr_solver.R, dtype=torch.float32, device=device)

    X = torch.tensor(x0, dtype=torch.float32, device=device).repeat(N_samples, 1).unsqueeze(-1)
    total_cost = torch.zeros(N_samples, 1, 1, device=device)
    
    for i in range(N_steps):
        t_current = i * dt
        t_batch = torch.full((N_samples,), t_current, dtype=torch.float32, device=device)
        X_input = X.transpose(1, 2)
        
        a = lqr_solver.markov_control(t_batch, X_input)
        a_col = a.unsqueeze(-1)
        
        cost_X = torch.bmm(torch.bmm(X.transpose(1, 2), C.expand(N_samples, 2, 2)), X)
        cost_a = torch.bmm(torch.bmm(a_col.transpose(1, 2), D.expand(N_samples, 2, 2)), a_col)
        total_cost += (cost_X + cost_a) * dt
        
        dW = torch.randn(N_samples, 2, 1, device=device) * np.sqrt(dt)
        drift = torch.bmm(H.expand(N_samples, 2, 2), X) + torch.bmm(M.expand(N_samples, 2, 2), a_col)
        diffusion = torch.bmm(sigma.expand(N_samples, 2, 2), dW)
        X = X + drift * dt + diffusion
        
    term_cost = torch.bmm(torch.bmm(X.transpose(1, 2), R.expand(N_samples, 2, 2)), X)
    total_cost += term_cost
    return total_cost.mean().item()

# ==========================================
# 3. 画图与测试函数 (Exercise 1.2)
# ==========================================
def plot_convergence(lqr_solver, x0, T):
    t_0 = torch.tensor([0.0], dtype=torch.float32)
    x_0_tensor = torch.tensor([[x0]], dtype=torch.float32)
    v_true = lqr_solver.value_function(t_0, x_0_tensor).item()
    print(f"理论真实值 v(0, x_0): {v_true:.6f}")

    N_samples_fixed = 10**5
    N_steps_list = [1, 10, 50, 100, 500, 1000, 5000]
    errors_time = []
    
    print("\n--- 开始测试: 时间离散化收敛 ---")
    for n_step in N_steps_list:
        v_mc = run_mc_lqr(lqr_solver, x0, T, N_steps=n_step, N_samples=N_samples_fixed)
        err = abs(v_mc - v_true)
        errors_time.append(err)
        print(f"N_steps: {n_step:4d} | MC Value: {v_mc:.4f} | Error: {err:.6f}")

    N_steps_fixed = 5000
    N_samples_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    errors_samples = []

    print("\n--- 开始测试: 蒙特卡洛样本量收敛 ---")
    for n_samp in N_samples_list:
        v_mc = run_mc_lqr(lqr_solver, x0, T, N_steps=N_steps_fixed, N_samples=n_samp)
        err = abs(v_mc - v_true)
        errors_samples.append(err)
        print(f"N_samples: {n_samp:6d} | MC Value: {v_mc:.4f} | Error: {err:.6f}")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.loglog(N_steps_list, errors_time, 'o-', label='MC Error')
    ref_line_1 = [errors_time[1] * (N_steps_list[1] / n) for n in N_steps_list]
    plt.loglog(N_steps_list, ref_line_1, 'r--', label='Reference O(1/N_steps)')
    plt.title('Convergence of Time Discretization')
    plt.xlabel('Number of Time Steps')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.subplot(1, 2, 2)
    plt.loglog(N_samples_list, errors_samples, 's-', color='orange', label='MC Error')
    ref_line_2 = [errors_samples[2] * np.sqrt(N_samples_list[2] / n) for n in N_samples_list]
    plt.loglog(N_samples_list, ref_line_2, 'r--', label='Reference O(1/sqrt(N_samples))')
    plt.title('Convergence of Monte Carlo Sampling')
    plt.xlabel('Number of MC Samples')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    # 加上这两行代码，将图表保存为高分辨率的 PNG 图片
    plt.savefig('convergence_plot.png', dpi=300)
    print("图表已成功保存为当前目录下的 'convergence_plot.png'，请在左侧文件树中点击查看！")


# ==========================================
# 4. 主程序运行入口
# ==========================================
if __name__ == "__main__":
    H_mat = np.array([[0.1, 0.0], [0.0, 0.1]])
    M_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma_mat = np.array([[0.2, 0.0], [0.0, 0.2]])
    C_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    D_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    T_val = 1.0

    print("正在初始化 LQR 求解器并求解 Riccati ODE...")
    lqr_solver = LQR_Solver(H_mat, M_mat, sigma_mat, C_mat, D_mat, R_mat, T_val)
    time_grid = np.linspace(0, T_val, 500)
    lqr_solver.solve_riccati(time_grid)
    print("Riccati ODE 求解完成！")

    x0_val = [1.0, -1.0] 
    print("开始运行蒙特卡洛验证，这需要占用一些算力计算，请稍候...")
    plot_convergence(lqr_solver, x0_val, T_val)