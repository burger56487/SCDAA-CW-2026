import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 导入你写好的 LQR_Solver 作为最终的测试基准
from ex1_mc import LQR_Solver

# ==========================================
# 1. 神经网络结构
# ==========================================
class NetDGM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super(NetDGM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class NetFFN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super(NetFFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc3(self.activation(self.fc2(self.activation(self.fc1(x)))))

# ==========================================
# 2. 损失函数
# ==========================================
def pde_loss(model_v, model_a, t, x, H, M, sigma, C, D, R_mat, T):
    """策略评估：给定策略 a，求解 V 的 DGM Loss"""
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    inputs = torch.cat([t, x], dim=1)
    u = model_v(inputs)
    
    du = torch.autograd.grad(u, inputs=[t, x], grad_outputs=torch.ones_like(u), create_graph=True)
    u_t, u_x = du[0], du[1]
    
    sigma_sq = sigma @ sigma.T
    trace_term = torch.zeros_like(u_t)
    for i in range(2):
        u_xi = u_x[:, i:i+1]
        du_xi = torch.autograd.grad(u_xi, x, grad_outputs=torch.ones_like(u_xi), create_graph=True)[0]
        for j in range(2):
            trace_term += 0.5 * sigma_sq[i, j] * du_xi[:, j:j+1]
            
    batch_size = t.shape[0]
    u_x_row = u_x.unsqueeze(1)
    x_col = x.unsqueeze(-1)
    term_Hx = torch.bmm(torch.bmm(u_x_row, H.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    
    # 获取当前的策略网络预测值 (无需梯度)
    with torch.no_grad():
        a_pred = model_a(inputs)
    a_col = a_pred.unsqueeze(-1)
    
    term_Ma = torch.bmm(torch.bmm(u_x_row, M.expand(batch_size, 2, 2)), a_col).squeeze(-1)
    
    x_row = x.unsqueeze(1)
    term_Cx = torch.bmm(torch.bmm(x_row, C.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    
    a_row = a_col.transpose(1, 2)
    term_Da = torch.bmm(torch.bmm(a_row, D.expand(batch_size, 2, 2)), a_col).squeeze(-1)
    
    residual = u_t + trace_term + term_Hx + term_Ma + term_Cx + term_Da
    loss_eqn = torch.mean(residual**2)
    
    t_T = torch.full_like(t, T, requires_grad=False)
    x_T = x.detach().clone()
    u_T_pred = model_v(torch.cat([t_T, x_T], dim=1))
    x_T_row = x_T.unsqueeze(1)
    x_T_col = x_T.unsqueeze(-1)
    u_T_true = torch.bmm(torch.bmm(x_T_row, R_mat.expand(batch_size, 2, 2)), x_T_col).squeeze(-1)
    loss_bound = torch.mean((u_T_pred - u_T_true)**2)
    
    return loss_eqn + loss_bound

def hamiltonian_loss(model_v, model_a, t, x, H, M, C, D):
    """策略改进：给定 V，最小化哈密顿量更新 a"""
    t.requires_grad_(True)
    x.requires_grad_(True)
    inputs = torch.cat([t, x], dim=1)
    
    # 1. 计算 \partial_x v (固定 v，不传播梯度到 model_v)
    u = model_v(inputs)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=False)[0]
    u_x = u_x.detach() # 切断价值函数的梯度图
    
    # 2. 计算当前网络输出的 a (开启梯度传播)
    a_pred = model_a(inputs)
    
    batch_size = t.shape[0]
    u_x_row = u_x.unsqueeze(1)
    x_col = x.unsqueeze(-1)
    a_col = a_pred.unsqueeze(-1)
    
    term_Hx = torch.bmm(torch.bmm(u_x_row, H.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    term_Ma = torch.bmm(torch.bmm(u_x_row, M.expand(batch_size, 2, 2)), a_col).squeeze(-1)
    
    x_row = x.unsqueeze(1)
    term_Cx = torch.bmm(torch.bmm(x_row, C.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    
    a_row = a_col.transpose(1, 2)
    term_Da = torch.bmm(torch.bmm(a_row, D.expand(batch_size, 2, 2)), a_col).squeeze(-1)
    
    # 哈密顿量 (无需取平均绝对误差，直接最小化其期望值)
    hamiltonian = term_Hx + term_Ma + term_Cx + term_Da
    return torch.mean(hamiltonian)

# ==========================================
# 3. 主程序：策略迭代训练
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前设备: {device}")

    # 系统矩阵
    H_mat = np.array([[0.1, 0.0], [0.0, 0.1]])
    M_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma_mat = np.array([[0.2, 0.0], [0.0, 0.2]])
    C_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    D_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    T_val = 1.0
    x0_val = [1.0, -1.0]

    H_t = torch.tensor(H_mat, dtype=torch.float32, device=device)
    M_t = torch.tensor(M_mat, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma_mat, dtype=torch.float32, device=device)
    C_t = torch.tensor(C_mat, dtype=torch.float32, device=device)
    D_t = torch.tensor(D_mat, dtype=torch.float32, device=device)
    R_t = torch.tensor(R_mat, dtype=torch.float32, device=device)

    # 真实的最优解基准
    lqr_solver = LQR_Solver(H_mat, M_mat, sigma_mat, C_mat, D_mat, R_mat, T_val)
    lqr_solver.solve_riccati(np.linspace(0, T_val, 500))
    test_t = torch.tensor([0.0], dtype=torch.float32, device=device)
    test_x = torch.tensor([[x0_val]], dtype=torch.float32, device=device)
    true_v = lqr_solver.value_function(test_t, test_x).item()
    true_a = lqr_solver.markov_control(test_t, test_x).detach().cpu().numpy()[0]

    model_v = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    model_a = NetFFN(input_dim=3, hidden_dim=100, output_dim=2).to(device)
    
    opt_v = optim.Adam(model_v.parameters(), lr=0.001)
    opt_a = optim.Adam(model_a.parameters(), lr=0.001)

    pi_iterations = 5   # 策略迭代的外部轮数
    inner_epochs = 800  # 内部训练网络轮数
    batch_size = 2000

    error_v_history = []
    error_a_history = []

    print(f"理论最优 v(0,x0): {true_v:.4f}")
    print(f"理论最优 a(0,x0): {true_a}")
    print("========================================")

    for pi_step in range(pi_iterations):
        print(f"\n---> 开始 Policy Iteration 第 {pi_step+1}/{pi_iterations} 轮")
        
        # 1. 策略评估 (更新 V)
        for _ in range(inner_epochs):
            opt_v.zero_grad()
            t_batch = torch.rand((batch_size, 1), dtype=torch.float32, device=device) * T_val
            x_batch = torch.rand((batch_size, 2), dtype=torch.float32, device=device) * 6.0 - 3.0
            loss_v = pde_loss(model_v, model_a, t_batch, x_batch, H_t, M_t, sigma_t, C_t, D_t, R_t, T_val)
            loss_v.backward()
            opt_v.step()
            
        # 2. 策略改进 (更新 a)
        for _ in range(inner_epochs):
            opt_a.zero_grad()
            t_batch = torch.rand((batch_size, 1), dtype=torch.float32, device=device) * T_val
            x_batch = torch.rand((batch_size, 2), dtype=torch.float32, device=device) * 6.0 - 3.0
            loss_ham = hamiltonian_loss(model_v, model_a, t_batch, x_batch, H_t, M_t, C_t, D_t)
            loss_ham.backward()
            opt_a.step()

        # 计算与真实 LQR 解的误差
        with torch.no_grad():
            t_eval = torch.tensor([[0.0]], dtype=torch.float32, device=device)
            x_eval = torch.tensor([x0_val], dtype=torch.float32, device=device)
            pred_v = model_v(torch.cat([t_eval, x_eval], dim=1)).item()
            pred_a = model_a(torch.cat([t_eval, x_eval], dim=1)).cpu().numpy()[0]
            
            err_v = abs(pred_v - true_v)
            err_a = np.linalg.norm(pred_a - true_a)
            error_v_history.append(err_v)
            error_a_history.append(err_a)
            
            print(f"评估完成 -> v预测: {pred_v:.4f} (误差 {err_v:.4f}) | a预测: {pred_a} (误差 {err_a:.4f})")

    # 绘制最终的 PI 收敛图表
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, pi_iterations+1), error_v_history, 'b-o')
    plt.title('Absolute Error of Value Function over PI iterations')
    plt.xlabel('Policy Iteration Step')
    plt.ylabel('Error vs LQR Optimal')
    plt.grid(True, ls="--")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, pi_iterations+1), error_a_history, 'r-s')
    plt.title('L2 Error of Markov Control over PI iterations')
    plt.xlabel('Policy Iteration Step')
    plt.ylabel('Norm Error vs LQR Optimal')
    plt.grid(True, ls="--")

    plt.tight_layout()
    plt.savefig('policy_iteration_convergence.png', dpi=300)
    print("\n✅ PIA 训练完成！包含收敛情况的图像已保存为 'policy_iteration_convergence.png'。")