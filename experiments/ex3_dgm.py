import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 1. 神经网络结构 (复用之前定义的单隐藏层网络)
# ==========================================
class NetDGM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super(NetDGM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh() 
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return out

# ==========================================
# 2. DGM 损失函数定义 (PDE 残差 + 边界条件)
# ==========================================
def dgm_loss(model, t, x, H, M, sigma, C, D, R_mat, T):
    # ---------------- 1. 内部方程残差 R_eqn ----------------
    # 必须开启梯度追踪以计算高阶导数
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    inputs = torch.cat([t, x], dim=1)
    u = model(inputs) # 输出 shape: (Batch, 1)
    
    # 计算一阶导数 \partial_t u 和 \partial_x u
    du = torch.autograd.grad(u, inputs=[t, x], grad_outputs=torch.ones_like(u), create_graph=True)
    u_t = du[0] # (Batch, 1)
    u_x = du[1] # (Batch, 2)
    
    # 计算二阶导数项 0.5 * tr(\sigma \sigma^T \partial_{xx} u)
    sigma_sq = sigma @ sigma.T
    trace_term = torch.zeros_like(u_t)
    for i in range(2):
        u_xi = u_x[:, i:i+1]
        du_xi = torch.autograd.grad(u_xi, x, grad_outputs=torch.ones_like(u_xi), create_graph=True)[0]
        for j in range(2):
            trace_term += 0.5 * sigma_sq[i, j] * du_xi[:, j:j+1]
            
    # 准备批量的矩阵乘法运算
    batch_size = t.shape[0]
    u_x_row = u_x.unsqueeze(1) # (B, 1, 2)
    x_col = x.unsqueeze(-1)    # (B, 2, 1)
    
    # (\partial_x u)^T H x
    term_Hx = torch.bmm(torch.bmm(u_x_row, H.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    
    # (\partial_x u)^T M \alpha (其中 \alpha = [1, 1]^T)
    alpha = torch.ones(batch_size, 2, 1, device=t.device)
    term_Malpha = torch.bmm(torch.bmm(u_x_row, M.expand(batch_size, 2, 2)), alpha).squeeze(-1)
    
    # x^T C x
    x_row = x.unsqueeze(1)
    term_Cx = torch.bmm(torch.bmm(x_row, C.expand(batch_size, 2, 2)), x_col).squeeze(-1)
    
    # \alpha^T D \alpha
    alpha_row = alpha.transpose(1, 2)
    term_Dalpha = torch.bmm(torch.bmm(alpha_row, D.expand(batch_size, 2, 2)), alpha).squeeze(-1)
    
    # PDE 残差
    residual = u_t + trace_term + term_Hx + term_Malpha + term_Cx + term_Dalpha
    loss_eqn = torch.mean(residual**2)
    
    # ---------------- 2. 边界条件残差 R_boundary ----------------
    t_T = torch.full_like(t, T, requires_grad=False)
    x_T = x.detach().clone() # 使用相同的空间点来评估边界
    
    u_T_pred = model(torch.cat([t_T, x_T], dim=1))
    
    x_T_row = x_T.unsqueeze(1)
    x_T_col = x_T.unsqueeze(-1)
    u_T_true = torch.bmm(torch.bmm(x_T_row, R_mat.expand(batch_size, 2, 2)), x_T_col).squeeze(-1)
    
    loss_bound = torch.mean((u_T_pred - u_T_true)**2)
    
    # 总损失
    return loss_eqn + loss_bound, loss_eqn.item(), loss_bound.item()

# ==========================================
# 3. 蒙特卡洛对照组 (固定策略 \alpha=[1,1]^T)
# ==========================================
def run_mc_constant_alpha(x0, T, N_steps, N_samples, H_np, M_np, sigma_np, C_np, D_np, R_np):
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
    
    # 根据题目要求，控制策略恒定为 (1,1)^T
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
# 4. 主程序：DGM 训练与 MC 验证
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的计算设备: {device}")

    # 系统矩阵
    H_mat = np.array([[0.1, 0.0], [0.0, 0.1]])
    M_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma_mat = np.array([[0.2, 0.0], [0.0, 0.2]])
    C_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    D_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    T_val = 1.0
    x0_val = [1.0, -1.0]

    # 将 numpy 矩阵转为 PyTorch Tensors 供 DGM 使用
    H_t = torch.tensor(H_mat, dtype=torch.float32, device=device)
    M_t = torch.tensor(M_mat, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma_mat, dtype=torch.float32, device=device)
    C_t = torch.tensor(C_mat, dtype=torch.float32, device=device)
    D_t = torch.tensor(D_mat, dtype=torch.float32, device=device)
    R_t = torch.tensor(R_mat, dtype=torch.float32, device=device)

    # 预计算用于对照的 MC 真实值
    print("正在通过蒙特卡洛计算线性 PDE 的真实基准值，请稍候...")
    mc_true_val = run_mc_constant_alpha(x0_val, T_val, N_steps=1000, N_samples=100000, 
                                        H_np=H_mat, M_np=M_mat, sigma_np=sigma_mat, 
                                        C_np=C_mat, D_np=D_mat, R_np=R_mat)
    print(f"MC 真实值 u(0, x0): {mc_true_val:.6f}")

    # 初始化 DGM 模型和优化器
    model = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3000
    batch_size = 2000
    history_loss = []
    history_mc_error = []
    test_t = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    test_x = torch.tensor([x0_val], dtype=torch.float32, device=device)

    print("开始训练 DGM...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 在域内随机采样点
        t_batch = torch.rand((batch_size, 1), dtype=torch.float32, device=device) * T_val
        x_batch = torch.rand((batch_size, 2), dtype=torch.float32, device=device) * 6.0 - 3.0
        
        loss, l_eqn, l_bound = dgm_loss(model, t_batch, x_batch, H_t, M_t, sigma_t, C_t, D_t, R_t, T_val)
        
        loss.backward()
        optimizer.step()
        
        history_loss.append(loss.item())
        
        # 定期检查与 MC 的误差
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                u_pred = model(torch.cat([test_t, test_x], dim=1)).item()
                mc_error = abs(u_pred - mc_true_val)
                history_mc_error.append(mc_error)
            
            if (epoch + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.6f} | u(0,x0) 预测: {u_pred:.4f} | MC 误差: {mc_error:.6f}")

    # 绘制结果
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_loss, label='Total DGM Loss')
    plt.yscale('log')
    plt.title('DGM Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.subplot(1, 2, 2)
    # x 轴为 epoch 的索引 (每 100 次记录一次)
    plt.plot(range(100, epochs+1, 100), history_mc_error, 'r-o', label='Absolute Error vs MC')
    plt.yscale('log')
    plt.title('DGM Error Evaluated against Monte Carlo')
    plt.xlabel('Epochs')
    plt.ylabel('Absolute Error')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.tight_layout()
    plt.savefig('dgm_linear_pde.png', dpi=300)
    print("DGM 训练完成！图表已保存为 'dgm_linear_pde.png'。")