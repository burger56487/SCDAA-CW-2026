import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 导入你的 LQR_Solver (请确保 ex1_mc.py 和此文件在同一目录下)
from ex1_mc import LQR_Solver

# ==========================================
# 1. 定义网络结构
# ==========================================
# Exercise 2.1: 单隐藏层网络，用于逼近价值函数 v(t,x)
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

# Exercise 2.2: 双隐藏层网络，用于逼近马尔可夫控制 a(t,x)
class NetFFN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super(NetFFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 第二个隐藏层
        self.activation = nn.Tanh() 
        self.fc3 = nn.Linear(hidden_dim, output_dim) # 输出维度为 2

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        return out

# ==========================================
# 2. 生成两组训练数据
# ==========================================
def generate_training_data(lqr_solver, N_data, T):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 采样 t 在 [0, T] 均匀分布
    t_batch = torch.rand(N_data, dtype=torch.float32, device=device) * T
    # 采样 x 在 [-3, 3] x [-3, 3] 均匀分布
    x_batch_flat = torch.rand((N_data, 2), dtype=torch.float32, device=device) * 6.0 - 3.0
    x_batch = x_batch_flat.unsqueeze(1) # shape: (N_data, 1, 2)
    
    # 拼接为网络输入特征 (t, x_1, x_2) -> shape: (N_data, 3)
    inputs = torch.cat([t_batch.unsqueeze(1), x_batch_flat], dim=1)
    
    # 无梯度计算真实标签
    with torch.no_grad():
        v_true = lqr_solver.value_function(t_batch, x_batch)    # 输出 shape: (N_data, 1)
        a_true = lqr_solver.markov_control(t_batch, x_batch)    # 输出 shape: (N_data, 2)
        
    return inputs, v_true, a_true

# ==========================================
# 3. 主训练流程
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的计算设备: {device}")

    # ---------------- 配置 LQR 参数并求解 ----------------
    H_mat = np.array([[0.1, 0.0], [0.0, 0.1]])
    M_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma_mat = np.array([[0.2, 0.0], [0.0, 0.2]])
    C_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    D_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    R_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    T_val = 1.0

    print("正在初始化基础 LQR 求解器提供真实标签...")
    lqr_solver = LQR_Solver(H_mat, M_mat, sigma_mat, C_mat, D_mat, R_mat, T_val)
    time_grid = np.linspace(0, T_val, 500)
    lqr_solver.solve_riccati(time_grid)

    # ---------------- 准备数据 ----------------
    N_data = 50000 
    print(f"正在生成 {N_data} 条训练数据...")
    inputs, v_targets, a_targets = generate_training_data(lqr_solver, N_data, T_val)

    # ---------------- 实例化网络和优化器 ----------------
    model_v = NetDGM(input_dim=3, hidden_dim=100, output_dim=1).to(device)
    model_a = NetFFN(input_dim=3, hidden_dim=100, output_dim=2).to(device) # Exercise 2.2 网络
    
    optimizer_v = optim.Adam(model_v.parameters(), lr=0.01)
    optimizer_a = optim.Adam(model_a.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # ---------------- 训练循环 ----------------
    epochs = 2000
    loss_history_v = []
    loss_history_a = []
    
    print("开始并行训练两个神经网络...")
    for epoch in range(epochs):
        # 训练价值函数网络 v(t,x)
        optimizer_v.zero_grad()
        v_pred = model_v(inputs)
        loss_v = criterion(v_pred, v_targets)
        loss_v.backward()
        optimizer_v.step()
        loss_history_v.append(loss_v.item())
        
        # 训练马尔可夫控制网络 a(t,x)
        optimizer_a.zero_grad()
        a_pred = model_a(inputs)
        loss_a = criterion(a_pred, a_targets)
        loss_a.backward()
        optimizer_a.step()
        loss_history_a.append(loss_a.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss V: {loss_v.item():.6f} | Loss A: {loss_a.item():.6f}")

    # ---------------- 绘制并保存损失图表 ----------------
    plt.figure(figsize=(14, 6))

    # 图 1: 价值函数的训练损失 (Exercise 2.1)
    plt.subplot(1, 2, 1)
    plt.plot(loss_history_v, color='blue', label='Value Function Loss (MSE)')
    plt.yscale('log')
    plt.title('Training Loss for Value Function (NetDGM)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # 图 2: 马尔可夫控制的训练损失 (Exercise 2.2)
    plt.subplot(1, 2, 2)
    plt.plot(loss_history_a, color='red', label='Markov Control Loss (MSE)')
    plt.yscale('log')
    plt.title('Training Loss for Markov Control (NetFFN)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('loss_plot_supervised_learning.png', dpi=300)
    print("\n✅ 训练完成！包含两张图表的图像已保存为 'loss_plot_supervised_learning.png'。")