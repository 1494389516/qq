"""
RL-S2V 简化版 - 核心机制演示
展示：状态 → 动作 → 奖励 → 更新策略 的循环

核心思想：
- 不需要梯度（黑盒）
- 用Q-learning学习攻击策略
- 通过试错找到最优攻击方法

核心优势（vs 梯度估算方法如ReWatt）：
✅ 查询高效：每步只查询1次，而梯度估算需要查询100+次
✅ 学习策略：通过Q网络学习，不需要每次都评估所有候选
✅ 实战可行：查询少，不易被检测系统发现

简化说明：
- 用简单的MLP代替Structure2Vec
- 只演示核心循环机制
- 重点理解强化学习思想和查询效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
import random

print("=" * 80)
print("🎮 RL-S2V 简化版 - 黑盒攻击核心机制")
print("=" * 80)
print()


# ============================================================================
# 第一部分：目标GCN模型（被攻击的黑盒模型）
# ============================================================================

class TargetGCN(nn.Module):
    """目标GCN模型 - 我们只能查询它，看不到内部"""
    
    def __init__(self, num_features, num_classes, hidden=16):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ============================================================================
# 第二部分：简化的Q网络
# ============================================================================

class SimpleQNetwork(nn.Module):
    """
    简化的Q网络：评估"添加这条边有多好"
    
    输入：边的两个端点特征
    输出：这条边的Q值（期望奖励）
    """
    
    def __init__(self, node_feature_dim, hidden=32):
        super().__init__()
        # 简单的MLP
        self.fc1 = nn.Linear(node_feature_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)  # 输出Q值
        
    def forward(self, node_features, edge):
        """
        计算添加边edge的Q值
        edge: [src, dst] 要添加的边
        """
        src, dst = edge
        # 拼接两个端点的特征
        edge_feat = torch.cat([node_features[src], node_features[dst]])
        
        # MLP计算Q值
        x = F.relu(self.fc1(edge_feat))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value


# ============================================================================
# 第三部分：RL攻击智能体
# ============================================================================

class SimpleRLAgent:
    """简化的强化学习攻击智能体"""
    
    def __init__(self, num_features, device='cpu'):
        self.device = device
        self.q_net = SimpleQNetwork(num_features).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.01)
        
        # 超参数
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.5  # 探索率（50%探索，50%利用）
        
    def select_action(self, node_features, candidate_edges):
        """
        ε-贪心策略选择动作
        
        50%概率：随机探索
        50%概率：选Q值最大的边
        """
        if random.random() < self.epsilon:
            # 探索：随机选一条边
            return random.choice(candidate_edges)
        else:
            # 利用：选Q值最大的边
            with torch.no_grad():
                q_values = []
                for edge in candidate_edges:
                    q = self.q_net(node_features, edge)
                    q_values.append(q.item())
                
                best_idx = np.argmax(q_values)
                return candidate_edges[best_idx]
    
    def update(self, state, action, reward, next_state):
        """
        更新Q网络
        
        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        """
        node_features = state
        
        # 当前Q值
        q_current = self.q_net(node_features, action)
        
        # 目标Q值：r + γ·max Q(s',a')
        with torch.no_grad():
            # 简化：假设下一步不再行动，Q值为0
            q_target = reward
        
        # 损失
        loss = F.mse_loss(q_current, torch.tensor([[q_target]], device=self.device))
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# ============================================================================
# 第四部分：攻击环境
# ============================================================================

class AttackEnvironment:
    """攻击环境：管理攻击过程"""
    
    def __init__(self, data, target_model, target_node, device='cpu'):
        self.data = data
        self.target_model = target_model
        self.target_node = target_node
        self.device = device
        
        # 原始预测
        self.true_label = data.y[target_node].item()
        
        # 当前状态
        self.reset()
        
    def reset(self):
        """重置环境"""
        self.edge_index = self.data.edge_index.clone()
        self.num_mods = 0
        self.query_count = 0  # 查询次数统计
        return self.data.x
    
    def get_candidates(self, budget=20):
        """获取候选边（简化版：随机采样）"""
        candidates = []
        num_nodes = self.data.num_nodes
        
        # 已有的边
        existing = set()
        for i in range(self.edge_index.size(1)):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            existing.add((min(src, dst), max(src, dst)))
        
        # 随机生成候选边
        while len(candidates) < budget:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v:
                edge = (min(u, v), max(u, v))
                if edge not in existing and edge not in candidates:
                    candidates.append([u, v])
        
        return candidates
    
    def step(self, action):
        """
        执行动作：添加一条边
        
        返回：(next_state, reward, done)
        """
        src, dst = action
        
        # 添加边（双向）
        new_edge = torch.tensor([[src, dst], [dst, src]], 
                               dtype=torch.long, device=self.device)
        self.edge_index = torch.cat([self.edge_index, new_edge], dim=1)
        self.num_mods += 1
        
        # ⚡ 关键：每步只查询1次黑盒模型（查询高效！）
        self.target_model.eval()
        with torch.no_grad():
            output = self.target_model(self.data.x, self.edge_index)
            pred = output[self.target_node].argmax().item()
        self.query_count += 1  # 统计查询次数
        
        # 计算奖励
        if pred != self.true_label:
            reward = 10.0  # 攻击成功！
            done = True
        elif self.num_mods >= 5:
            reward = -5.0  # 达到预算，失败
            done = True
        else:
            reward = -0.1  # 继续尝试
            done = False
        
        next_state = self.data.x
        
        return next_state, reward, done, pred


# ============================================================================
# 第五部分：主程序 - 展示完整循环
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 设备: {device}\n")
    
    # 1. 加载数据
    print("📂 加载Cora数据集...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    print(f"   节点: {data.num_nodes}, 边: {data.num_edges}\n")
    
    # 2. 训练目标模型（黑盒）
    print("🎯 训练目标GCN（黑盒模型）...")
    target_model = TargetGCN(data.num_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
    
    target_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = target_model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    
    # 测试
    target_model.eval()
    with torch.no_grad():
        pred = target_model(data.x, data.edge_index).argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    print(f"   准确率: {acc:.4f}\n")
    
    # 3. 选择攻击目标
    test_nodes = torch.where(data.test_mask)[0]
    correct = test_nodes[pred[test_nodes] == data.y[test_nodes]]
    target_node = correct[0].item()
    
    print(f"🎯 攻击目标节点: {target_node}")
    print(f"   真实标签: {data.y[target_node].item()}")
    print(f"   当前预测: {pred[target_node].item()}\n")
    
    # 4. 创建RL智能体
    print("🤖 创建RL攻击智能体...\n")
    agent = SimpleRLAgent(data.num_features, device)
    
    # 5. 训练智能体（核心循环！）
    print("=" * 80)
    print("🏋️  开始训练攻击策略（观察核心循环）")
    print("=" * 80)
    print()
    
    num_episodes = 10  # 简化版：只训练10轮
    
    for episode in range(num_episodes):
        print(f"📍 Episode {episode + 1}/{num_episodes}")
        print("-" * 60)
        
        # 创建环境
        env = AttackEnvironment(data, target_model, target_node, device)
        state = env.reset()
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 5:
            # 🔵 步骤1：获取候选动作
            candidates = env.get_candidates(budget=10)
            
            # 🔵 步骤2：智能体选择动作（ε-贪心）
            action = agent.select_action(state, candidates)
            
            # 🔵 步骤3：执行动作，观察结果
            next_state, reward, done, current_pred = env.step(action)
            
            # 🔵 步骤4：更新Q网络（学习！）
            loss = agent.update(state, action, reward, next_state)
            
            episode_reward += reward
            
            print(f"   步骤{step + 1}: 添加边{action} → "
                  f"预测={current_pred} | 奖励={reward:.1f} | "
                  f"Q-loss={loss:.4f}")
            
            state = next_state
            step += 1
        
        print(f"   总奖励: {episode_reward:.1f}")
        
        if episode_reward > 0:
            print("   ✅ 攻击成功！")
        else:
            print("   ❌ 攻击失败")
        print()
    
    # 6. 测试学到的策略
    print("=" * 80)
    print("🚀 用学到的策略进行最终攻击")
    print("=" * 80)
    print()
    
    agent.epsilon = 0  # 关闭探索，纯利用
    env = AttackEnvironment(data, target_model, target_node, device)
    state = env.reset()
    
    done = False
    step = 0
    modifications = []
    
    while not done and step < 5:
        candidates = env.get_candidates(budget=10)
        action = agent.select_action(state, candidates)
        next_state, reward, done, current_pred = env.step(action)
        
        modifications.append(action)
        
        print(f"步骤{step + 1}:")
        print(f"  添加边: {action}")
        print(f"  当前预测: {current_pred}")
        print(f"  奖励: {reward:.1f}")
        
        if done and reward > 0:
            print(f"\n🎉 攻击成功！")
            print(f"   修改边数: {len(modifications)} 条")
            print(f"   查询次数: {env.query_count} 次 ⚡")
            print(f"   预测从 {data.y[target_node].item()} → {current_pred}")
            print(f"   平均每步查询: {env.query_count / len(modifications):.1f} 次")
            break
        
        state = next_state
        step += 1
    
    if not done or reward < 0:
        print(f"\n❌ 攻击失败（预算不够或策略未学好）")
        print(f"   总查询次数: {env.query_count}")
    
    print("\n" + "=" * 80)
    print("💡 核心观察 - RL-S2V的优势")
    print("=" * 80)
    print("""
1. ⚡ 查询效率（最大优势）：
   - 每步只查询1次黑盒模型
   - 对比：梯度估算方法需要查询100+次/步
   - 实战意义：查询少 = 不易被检测 = 更可行
   
2. 🔄 核心循环：
   状态观察 → 选择动作 → 获得奖励 → 更新Q网络 → 重复
   
3. 🎮 探索vs利用：
   - 训练时：50%探索（随机），50%利用（Q值最大）
   - 测试时：100%利用（只选最佳）
   
4. 🧠 学习能力：
   - Q网络逐渐学会"哪种边最有效"
   - 不需要每次都评估所有候选边
   
5. 🕵️ 黑盒特性：
   - 只查询模型输出（预测结果）
   - 不需要梯度
   - 更接近真实攻击场景
   
6. 📊 效率对比：
   ┌─────────────┬────────────┬────────────┐
   │   方法      │ 每步查询   │  适用场景  │
   ├─────────────┼────────────┼────────────┤
   │ RL-S2V      │ 1次 ⚡     │ 实战       │
   │ 梯度估算    │ 100+次     │ 理论研究   │
   └─────────────┴────────────┴────────────┘
    """)
    print("=" * 80)


if __name__ == '__main__':
    main()

