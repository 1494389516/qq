"""
简单的图对抗算法实现
包括：
1. 简单的图神经网络（GCN）
2. 基于梯度的对抗攻击（FGSM）
3. 随机边扰动攻击
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import copy


class SimpleGCN(nn.Module):
    """简单的图卷积网络"""
    
    def __init__(self, num_features, num_classes, hidden_dim=16):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        
    def forward(self, x, edge_index):
        # 第一层GCN + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphAdversarialAttack:
    """图对抗攻击类"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def fgsm_attack(self, data, target_node, epsilon=0.1):
        """
        FGSM (Fast Gradient Sign Method) 对抗攻击
        对节点特征进行对抗扰动
        
        参数:
            data: 图数据
            target_node: 目标节点索引
            epsilon: 扰动强度
        """
        # 复制数据
        adv_data = copy.deepcopy(data)
        adv_data.x.requires_grad = True
        
        # 前向传播
        self.model.eval()
        output = self.model(adv_data.x, adv_data.edge_index)
        
        # 计算损失
        loss = F.nll_loss(output[target_node:target_node+1], 
                         adv_data.y[target_node:target_node+1])
        
        # 反向传播获取梯度
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗样本
        data_grad = adv_data.x.grad.data
        perturbed_data = adv_data.x + epsilon * data_grad.sign()
        
        # 创建对抗图数据
        adv_data.x = perturbed_data.detach()
        
        return adv_data
    
    def random_edge_attack(self, data, num_perturbations=5, add_edges=True):
        """
        随机边扰动攻击
        随机添加或删除边
        
        参数:
            data: 图数据
            num_perturbations: 扰动的边数量
            add_edges: True为添加边，False为删除边
        """
        adv_data = copy.deepcopy(data)
        edge_index = adv_data.edge_index.cpu().numpy()
        num_nodes = adv_data.x.shape[0]
        
        if add_edges:
            # 随机添加边
            new_edges = []
            for _ in range(num_perturbations):
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
                if src != dst:  # 避免自环
                    new_edges.append([src, dst])
                    new_edges.append([dst, src])  # 无向图
            
            if new_edges:
                new_edges = np.array(new_edges).T
                edge_index = np.concatenate([edge_index, new_edges], axis=1)
        else:
            # 随机删除边
            num_edges = edge_index.shape[1]
            if num_edges > num_perturbations * 2:
                keep_indices = np.random.choice(num_edges, 
                                              num_edges - num_perturbations * 2, 
                                              replace=False)
                edge_index = edge_index[:, keep_indices]
        
        adv_data.edge_index = torch.tensor(edge_index, dtype=torch.long)
        return adv_data
    
    def gradient_based_edge_attack(self, data, target_node, num_perturbations=5):
        """
        基于梯度的边攻击
        选择影响最大的边进行扰动
        
        参数:
            data: 图数据
            target_node: 目标节点
            num_perturbations: 扰动数量
        """
        # 简化版本：使用节点梯度估计重要性
        adv_data = copy.deepcopy(data)
        adv_data.x.requires_grad = True
        
        self.model.eval()
        output = self.model(adv_data.x, adv_data.edge_index)
        loss = F.nll_loss(output[target_node:target_node+1], 
                         adv_data.y[target_node:target_node+1])
        
        self.model.zero_grad()
        loss.backward()
        
        # 计算每个节点的重要性（基于梯度范数）
        node_importance = torch.norm(adv_data.x.grad, dim=1, p=2).cpu().numpy()
        
        # 选择最重要的节点，在它们之间添加边
        important_nodes = np.argsort(node_importance)[-num_perturbations*2:]
        
        edge_index = adv_data.edge_index.cpu().numpy()
        new_edges = []
        
        for i in range(len(important_nodes)-1):
            src = important_nodes[i]
            dst = important_nodes[i+1]
            new_edges.append([src, dst])
            new_edges.append([dst, src])
        
        if new_edges:
            new_edges = np.array(new_edges).T
            edge_index = np.concatenate([edge_index, new_edges], axis=1)
        
        adv_data.edge_index = torch.tensor(edge_index, dtype=torch.long)
        return adv_data


def train_model(model, data, epochs=200, lr=0.01):
    """训练GCN模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            _, pred = model(data.x, data.edge_index).max(dim=1)
            correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            acc = correct / data.test_mask.sum().item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}')
            model.train()


def evaluate_model(model, data):
    """评估模型准确率"""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
    return acc


def main():
    """主函数：演示图对抗攻击"""
    print("=" * 50)
    print("图对抗算法演示")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载Cora数据集
    print("\n正在加载Cora数据集...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    print(f"数据集信息:")
    print(f"  - 节点数: {data.num_nodes}")
    print(f"  - 边数: {data.num_edges}")
    print(f"  - 特征维度: {dataset.num_features}")
    print(f"  - 类别数: {dataset.num_classes}")
    
    # 创建并训练模型
    print("\n正在训练GCN模型...")
    model = SimpleGCN(dataset.num_features, dataset.num_classes).to(device)
    train_model(model, data, epochs=200)
    
    # 评估原始模型
    original_acc = evaluate_model(model, data)
    print(f"\n原始模型测试准确率: {original_acc:.4f}")
    
    # 创建对抗攻击器
    attacker = GraphAdversarialAttack(model, device)
    
    # 1. FGSM特征攻击
    print("\n" + "=" * 50)
    print("1. FGSM特征对抗攻击")
    print("=" * 50)
    target_node = data.test_mask.nonzero()[0].item()
    
    # 改进：测试更多epsilon值
    epsilons = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    print(f"\n{'Epsilon':<10} {'准确率':<10} {'下降':<10} {'下降%':<10} {'效果'}")
    print("-" * 60)
    
    for epsilon in epsilons:
        adv_data_fgsm = attacker.fgsm_attack(data, target_node, epsilon=epsilon)
        adv_acc = evaluate_model(model, adv_data_fgsm)
        drop = original_acc - adv_acc
        drop_pct = (drop / original_acc) * 100
        
        # 效果评价
        if drop < 0.02:
            effect = "😐 无效"
        elif drop < 0.05:
            effect = "😊 轻微"
        elif drop < 0.10:
            effect = "😟 中等"
        elif drop < 0.15:
            effect = "😱 显著"
        else:
            effect = "💀 严重"
        
        print(f"{epsilon:<10.2f} {adv_acc:<10.4f} {drop:<10.4f} {drop_pct:<10.2f} {effect}")
    
    # 2. 随机边攻击
    print("\n" + "=" * 50)
    print("2. 随机边扰动攻击")
    print("=" * 50)
    
    for num_perturb in [10, 20, 50]:
        adv_data_edge = attacker.random_edge_attack(data, num_perturbations=num_perturb)
        adv_acc = evaluate_model(model, adv_data_edge)
        print(f"添加{num_perturb}条边: 攻击后准确率 = {adv_acc:.4f} "
              f"(下降 {(original_acc - adv_acc):.4f})")
    
    # 3. 基于梯度的边攻击
    print("\n" + "=" * 50)
    print("3. 基于梯度的边攻击")
    print("=" * 50)
    
    for num_perturb in [5, 10, 20]:
        adv_data_grad = attacker.gradient_based_edge_attack(data, target_node, 
                                                            num_perturbations=num_perturb)
        adv_acc = evaluate_model(model, adv_data_grad)
        print(f"添加{num_perturb}条边: 攻击后准确率 = {adv_acc:.4f} "
              f"(下降 {(original_acc - adv_acc):.4f})")
    
    print("\n" + "=" * 50)
    print("对抗攻击演示完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
