"""
ReWatt (AAAI 2021) - 查询高效的黑盒攻击

核心思想：
- 在黑盒条件下，通过少量查询"估算"出梯度。
- 用白盒攻击的思路（梯度下降），解决黑盒攻击的问题。
- 极大地降低了攻击所需的查询次数，更接近实战。

简化说明：
- 我们将去掉RL-S2V的Q网络和强化学习循环。
- 核心替换为"梯度估算"和"贪心选择"模块。
- 重点理解如何用有限差分来高效地指导攻击。

正确的对比说明：
- 修正1：统一候选边生成策略（都只连接到目标节点）
- 修正2：强化RL-S2V的预热训练（50轮，充分学习攻击模式）
- 修正3：RL-S2V主循环只用Q网络选边，每步仅查询1次（体现查询高效）
- 核心区别：
  * ReWatt: 查询密集型（每步100+查询），精准选边，步数少
  * RL-S2V: 查询节约型（每步1查询），依赖学习，步数可能多
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
import random
import time


# ============================================================================
# 第一部分：目标GCN模型（被攻击的黑盒模型）
# (与之前相同)
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


def get_probas(model, data, target_node):
    """辅助函数：获取目标节点的概率分布"""
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        return torch.exp(output[target_node])

def compute_test_acc(model, data):
    """辅助函数：计算测试集准确率"""
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    return float(acc.item())


# ============================================================================
# 补充：简化版RL-S2V（用于与ReWatt对比）
# ============================================================================

class SimpleQNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(node_feature_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
    def forward(self, node_features, edge):
        src, dst = edge
        edge_feat = torch.cat([node_features[src], node_features[dst]])
        x = F.relu(self.fc1(edge_feat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimpleRLAgent:
    def __init__(self, num_features, device='cpu'):
        self.device = device
        self.q_net = SimpleQNetwork(num_features).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.01)
        self.epsilon = 0.1  # 修正：进一步降低随机性，提高攻击效果
    def select_action(self, node_features, candidate_edges):
        if random.random() < self.epsilon:
            return random.choice(candidate_edges)
        with torch.no_grad():
            q_values = []
            for edge in candidate_edges:
                q = self.q_net(node_features, edge)
                q_values.append(q.item())
            best_idx = int(np.argmax(q_values))
            return candidate_edges[best_idx]
    def update(self, state, action, reward):
        q_current = self.q_net(state, action)
        # 简化：目标Q值=即时奖励，完全修复维度匹配
        target = torch.tensor(reward, device=self.device).view_as(q_current)
        loss = F.mse_loss(q_current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

def get_random_candidate_edges(data, budget, target_node):
    candidates = []
    num_nodes = data.num_nodes
    existing = set()
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        existing.add(tuple(sorted((u, v))))
    
    # 修正：与ReWatt一致，只生成连接到目标节点的边，确保公平对比
    while len(candidates) < budget:
        u = random.randint(0, num_nodes - 1)
        v = target_node  # 修正：只连接到目标节点
        if u != v:
            e = tuple(sorted((u, v)))
            if e not in existing and list(e) not in candidates:
                candidates.append(list(e))
    return candidates

# ============================================================================
# 第二部分：ReWatt 攻击智能体
# (这是与RL-S2V最大的不同之处)
# ============================================================================

class ReWattAgent:
    """
    ReWatt攻击智能体：通过梯度估计进行攻击
    - 没有Q网络
    - 没有复杂的学习过程
    - 核心是高效的“试探”
    """
    
    def __init__(self, target_model, data, target_node, device='cpu'):
        self.target_model = target_model
        self.data = data
        self.target_node = target_node
        self.device = device
        self.query_count = 0

    def get_candidates(self, budget=100):
        """获取候选边（随机采样）"""
        candidates = []
        num_nodes = self.data.num_nodes
        
        # 获取已有的边，避免重复
        existing_edges = set()
        edge_index = self.data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[:, i]
            existing_edges.add(tuple(sorted((u, v))))

        # 随机生成不重复的候选边
        while len(candidates) < budget:
            u = random.randint(0, num_nodes - 1)
            v = self.target_node # 简化：只考虑连接到目标节点的边
            if u != v:
                edge = tuple(sorted((u, v)))
                if edge not in existing_edges and edge not in candidates:
                    candidates.append(edge)
        
        return [list(c) for c in candidates]

    def estimate_gradients(self, candidates, original_prob):
        """
        核心！用有限差分估计梯度（影响力）
        对每个候选边，进行两次查询来估算其影响力
        """
        influences = []
        
        for edge in candidates:
            src, dst = edge
            
            # 构造一个临时的新图（只添加一条边）
            temp_edge_index = self.data.edge_index.clone()
            new_edge = torch.tensor([[src, dst], [dst, src]], dtype=torch.long, device=self.device)
            temp_edge_index = torch.cat([temp_edge_index, new_edge], dim=1)
            
            # 查询模型，获取新概率
            self.target_model.eval()
            with torch.no_grad():
                output = self.target_model(self.data.x, temp_edge_index)
                prob = torch.exp(output)[self.target_node]
                self.query_count += 1
            
            # 计算影响力（损失的变化）
            true_label = self.data.y[self.target_node]
            new_prob = prob[true_label]
            influence = original_prob - new_prob.item() # 目标类别概率下降得越多，影响力越大
            influences.append(influence)
            
        return influences

    def select_action(self, candidates):
        """
        贪心选择：选择估算梯度（影响力）最大的动作
        """
        # 1. 获取原始概率
        self.target_model.eval()
        with torch.no_grad():
            output = self.target_model(self.data.x, self.data.edge_index)
            probs = torch.exp(output)
            true_label = self.data.y[self.target_node]
            original_prob = probs[self.target_node, true_label].item()
            self.query_count += 1

        # 2. 估算所有候选边的梯度
        influences = self.estimate_gradients(candidates, original_prob)
        
        # 3. 选择影响力最大的边
        best_idx = np.argmax(influences)
        best_edge = candidates[best_idx]
        
        return best_edge


# ============================================================================
# 第三部分：攻击主程序
# ============================================================================

def run_rewatt_attack(target_model, data, target_node, device):
    """执行ReWatt攻击并返回结果（无中间打印）"""
    
    # 创建一个新的data对象副本，避免修改原始数据
    attack_data = data.clone()
    
    agent = ReWattAgent(target_model, attack_data, target_node, device)
    
    attack_budget = 10 # 增加预算，确保攻击成功
    modifications = []
    
    start_time = time.time()
    
    for i in range(attack_budget):
        # 步骤1: 获取候选边
        candidates = agent.get_candidates(budget=100)
        # 步骤2: 选择最佳攻击边
        best_edge = agent.select_action(candidates)
        modifications.append(best_edge)
        # 步骤3: 应用攻击，更新图结构
        src, dst = best_edge
        new_edge_tensor = torch.tensor([[src, dst], [dst, src]], dtype=torch.long, device=device)
        agent.data.edge_index = torch.cat([agent.data.edge_index, new_edge_tensor], dim=1)
        # 步骤4: 检查攻击是否成功
        target_model.eval()
        with torch.no_grad():
            output = target_model(agent.data.x, agent.data.edge_index)
            current_pred = output[target_node].argmax().item()
            agent.query_count += 1
        if current_pred != data.y[target_node].item():
            break

    end_time = time.time()
    
    success = current_pred != data.y[target_node].item()
    
    return {
        "success": success,
        "queries": agent.query_count,
        "mods": len(modifications),
        "time": end_time - start_time,
        "modified_data": agent.data
    }

def run_rl_s2v_attack(target_model, data, target_node, device):
    """执行RL-S2V攻击 - 核心优势：查询次数少，依赖Q网络学习"""
    attack_data = data.clone()
    max_steps = 20  # 允许更多步数，因为每步查询少
    modifications = []
    query_count = 0
    start_time = time.time()

    agent = SimpleRLAgent(attack_data.num_features, device)
    node_features = attack_data.x

    # 充分的预热训练：让Q网络学习什么样的边有效
    for _ in range(50):  # 更多训练轮次
        candidates = get_random_candidate_edges(attack_data, budget=100, target_node=target_node)
        action = agent.select_action(node_features, candidates)
        src, dst = action
        temp_edge = torch.tensor([[src, dst], [dst, src]], dtype=torch.long, device=device)
        temp_edge_index = torch.cat([attack_data.edge_index, temp_edge], dim=1)
        target_model.eval()
        with torch.no_grad():
            out_before = target_model(attack_data.x, attack_data.edge_index)
            out_after = target_model(attack_data.x, temp_edge_index)
            # 计算logit差异作为奖励
            logit_true = out_before[target_node, data.y[target_node]]
            logit_true_after = out_after[target_node, data.y[target_node]]
            logit_drop = logit_true - logit_true_after
            reward = float(logit_drop.item() * 100)
        agent.update(node_features, action, reward)

    # 主攻击循环：关键是只用Q网络选边，不评估所有候选
    for step in range(max_steps):
        # 生成候选边
        candidates = get_random_candidate_edges(attack_data, budget=100, target_node=target_node)
        
        # RL-S2V核心：直接用Q网络选择，不查询黑盒模型评估所有候选
        # 只用Q网络评估，不消耗查询次数
        best_action = None
        best_q = float('-inf')
        for candidate in candidates:
            with torch.no_grad():
                q_value = agent.q_net(node_features, candidate).item()
            if q_value > best_q:
                best_q = q_value
                best_action = candidate
        
        # 应用选中的边
        action = best_action
        modifications.append(action)
        src, dst = action
        new_edge_tensor = torch.tensor([[src, dst], [dst, src]], dtype=torch.long, device=device)
        attack_data.edge_index = torch.cat([attack_data.edge_index, new_edge_tensor], dim=1)
        
        # 只查询一次：检查攻击是否成功
        target_model.eval()
        with torch.no_grad():
            output = target_model(attack_data.x, attack_data.edge_index)
            current_pred = output[target_node].argmax().item()
            current_logit = output[target_node, data.y[target_node]].item()
            query_count += 1
        
        # 计算奖励并更新Q网络
        if current_pred != data.y[target_node].item():
            reward = 100.0  # 攻击成功
            agent.update(node_features, action, reward)
            break
        else:
            # 没成功，根据logit变化给奖励继续学习
            reward = -current_logit * 10  # logit越低越好
            agent.update(node_features, action, reward)

    end_time = time.time()
    success = current_pred != data.y[target_node].item()

    return {
        "success": success,
        "queries": query_count,
        "mods": len(modifications),
        "time": end_time - start_time,
        "modified_data": attack_data
    }

def run_random_attack(target_model, data, target_node, device):
    """执行随机攻击作为基线，并返回结果（无中间打印）"""
    
    attack_data = data.clone()
    
    attack_budget = 5
    modifications = []
    query_count = 0
    start_time = time.time()
    
    # 获取候选边生成器
    agent = ReWattAgent(target_model, attack_data, target_node, device)

    for i in range(attack_budget):
        # 步骤1: 获取候选边
        candidates = agent.get_candidates(budget=100)
        # 步骤2: 随机选择一条边
        random_edge = random.choice(candidates)
        modifications.append(random_edge)
        # 步骤3: 应用攻击
        src, dst = random_edge
        new_edge_tensor = torch.tensor([[src, dst], [dst, src]], dtype=torch.long, device=device)
        attack_data.edge_index = torch.cat([attack_data.edge_index, new_edge_tensor], dim=1)
        # 步骤4: 检查是否成功
        target_model.eval()
        with torch.no_grad():
            output = target_model(attack_data.x, attack_data.edge_index)
            current_pred = output[target_node].argmax().item()
            query_count += 1
        if current_pred != data.y[target_node].item():
            break

    end_time = time.time()

    success = current_pred != data.y[target_node].item()
    
    return {
        "success": success,
        "queries": query_count,
        "mods": len(modifications),
        "time": end_time - start_time,
        "modified_data": attack_data
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. 加载数据
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    # 2. 训练目标模型（黑盒）
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
    test_acc_before = float(acc.item())
    
    # 3. 选择攻击目标（不打印中间信息）
    test_nodes = torch.where(data.test_mask)[0]
    correct_nodes = test_nodes[pred[test_nodes] == data.y[test_nodes]]
    if len(correct_nodes) == 0:
        return
    target_node = correct_nodes[0].item()
    true_label = data.y[target_node].item()

    # 4. 执行ReWatt攻击
    rewatt_results = run_rewatt_attack(target_model, data, target_node, device)
    # 计算ReWatt攻击后的测试集准确率
    test_acc_after_rewatt = compute_test_acc(target_model, rewatt_results["modified_data"])

    # 5. 执行RL-S2V攻击（简化版）
    rl_s2v_results = run_rl_s2v_attack(target_model, data, target_node, device)
    # 计算RL-S2V攻击后的测试集准确率
    test_acc_after_rl = compute_test_acc(target_model, rl_s2v_results["modified_data"])

    # 6. 结果总结与对比
    # 输出最终对比结果（中文，聚焦损失）
    print(f"初始测试集准确率: {test_acc_before:.4f}")
    print("=" * 70)
    print(f"目标节点: {target_node} | 真实标签: {true_label}")
    print("-" * 70)
    # ReWatt 结果
    rewatt_success = "✅成功" if rewatt_results['success'] else "❌失败"
    print(f"ReWatt  → {rewatt_success} | 步数:{rewatt_results['mods']} | 查询:{rewatt_results['queries']} | 准确率:{test_acc_after_rewatt:.4f} | 下降:{(test_acc_before - test_acc_after_rewatt)*100:.2f}pp")
    # RL-S2V 结果
    rl_success = "✅成功" if rl_s2v_results['success'] else "❌失败"
    print(f"RL-S2V → {rl_success} | 步数:{rl_s2v_results['mods']} | 查询:{rl_s2v_results['queries']} | 准确率:{test_acc_after_rl:.4f} | 下降:{(test_acc_before - test_acc_after_rl)*100:.2f}pp")
    print("=" * 70)
    print("说明：")
    print("- '成功' = 目标节点被成功误分类")
    print("- '步数' = 实际添加的边数量")
    print("- '查询' = 查询黑盒模型的总次数")
    print("- '下降' = 测试集准确率下降的百分点")
    print()
    print("算法对比（理论差异）：")
    print("┌─────────┬──────────────┬──────────────┬──────────────┐")
    print("│  算法   │  每步查询    │  总查询次数  │   优势       │")
    print("├─────────┼──────────────┼──────────────┼──────────────┤")
    print("│ ReWatt  │ ~100次/步    │  多（精准）  │ 步数少、准确 │")
    print("│ RL-S2V  │  1次/步      │  少（学习）  │ 查询高效     │")
    print("└─────────┴──────────────┴──────────────┴──────────────┘")
    print()
    print("ReWatt论文的核心创新：相比传统RL方法，用梯度估算减少查询")
    print("但这里为了对比清晰，ReWatt用了更多候选边评估")


if __name__ == '__main__':
    main()
