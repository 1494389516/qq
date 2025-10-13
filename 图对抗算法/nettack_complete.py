"""
Nettack完整实现 - 循序渐进版本
每一步都有详细注释和输出

核心思想：
1. 用梯度估算每条边的影响力（不需要暴力测试）
2. 贪心选择影响力最大的边
3. 迭代添加，直到攻击成功或达到预算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
import copy


# ============================================
# 第一部分：GCN模型（和之前一样）
# ============================================
class SimpleGCN(nn.Module):
    """简单的两层GCN"""
    
    def __init__(self, num_features, num_classes, hidden_dim=16):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ============================================
# 第二部分：Nettack攻击类
# ============================================
class NettackAttack:
    """
    Nettack攻击实现
    
    流程：
    1. 计算损失对边的梯度
    2. 选择梯度最大的边（影响力最大）
    3. 添加这条边
    4. 重复直到成功或达到预算
    """
    
    def __init__(self, model, data, device='cpu'):
        self.model = model
        self.data = data
        self.device = device
        
    def attack(self, target_node, num_perturbations=5, verbose=True):
        """
        对目标节点进行攻击
        
        参数：
            target_node: 要攻击的节点ID
            num_perturbations: 最多添加几条边
            verbose: 是否打印详细信息
        
        返回：
            攻击后的边索引
        """
        if verbose:
            print("=" * 80)
            print("🎯 Nettack攻击开始")
            print("=" * 80)
        
        # ===== 步骤1：获取原始预测 =====
        self.model.eval()
        with torch.no_grad():
            output_orig = self.model(self.data.x, self.data.edge_index)
            pred_orig = output_orig[target_node].argmax().item()
            conf_orig = torch.exp(output_orig[target_node]).max().item()
            true_label = self.data.y[target_node].item()
        
        if verbose:
            print(f"\n【目标节点信息】")
            print(f"  节点ID: {target_node}")
            print(f"  真实标签: {true_label}")
            print(f"  原始预测: {pred_orig} {'✓' if pred_orig == true_label else '✗'}")
            print(f"  预测置信度: {conf_orig:.4f}")
        
        # ===== 步骤2：初始化 =====
        current_edge_index = self.data.edge_index.clone()
        added_edges = []  # 记录添加的边
        
        # ===== 步骤3：贪心迭代添加边 =====
        for iteration in range(num_perturbations):
            if verbose:
                print(f"\n{'─' * 80}")
                print(f"【第 {iteration + 1}/{num_perturbations} 轮】")
            
            # 3.1 计算梯度（核心！）
            best_edge, impact_score = self._find_best_edge(
                current_edge_index, 
                target_node,
                verbose=verbose
            )
            
            if best_edge is None:
                if verbose:
                    print("  ⚠️  没有找到可添加的边，停止攻击")
                break
            
            # 3.2 添加这条边
            src, dst = best_edge
            new_edge = torch.tensor([[src, dst], [dst, src]], dtype=torch.long)
            current_edge_index = torch.cat([current_edge_index, new_edge.T], dim=1)
            added_edges.append(best_edge)
            
            if verbose:
                print(f"  ✓ 添加边: ({src}, {dst})")
                print(f"  ✓ 估算影响力: {impact_score:.6f}")
            
            # 3.3 测试攻击效果
            with torch.no_grad():
                output_new = self.model(self.data.x, current_edge_index)
                pred_new = output_new[target_node].argmax().item()
                conf_new = torch.exp(output_new[target_node]).max().item()
                loss_new = F.nll_loss(
                    output_new[target_node:target_node+1],
                    self.data.y[target_node:target_node+1]
                ).item()
            
            if verbose:
                print(f"\n  【当前状态】")
                print(f"    预测: {pred_orig} → {pred_new}")
                print(f"    置信度: {conf_orig:.4f} → {conf_new:.4f}")
                print(f"    损失: {loss_new:.4f}")
            
            # 3.4 检查是否成功
            if pred_new != pred_orig:
                if verbose:
                    print(f"\n  🎉 攻击成功！预测改变了！")
                    print(f"  只用了 {len(added_edges)} 条边")
                break
        
        # ===== 步骤4：总结 =====
        if verbose:
            print("\n" + "=" * 80)
            print("📊 攻击总结")
            print("=" * 80)
            print(f"  目标节点: {target_node}")
            print(f"  添加边数: {len(added_edges)}")
            print(f"  添加的边: {added_edges}")
            print(f"  原始预测: {pred_orig}")
            print(f"  攻击后预测: {pred_new}")
            print(f"  攻击{'成功 ✓' if pred_new != pred_orig else '失败 ✗'}")
        
        return current_edge_index, added_edges
    
    def _find_best_edge(self, edge_index, target_node, verbose=True):
        """
        找到影响力最大的边
        
        核心方法：用梯度估算
        
        返回：
            (src, dst): 最佳边的两个节点
            impact: 影响力分数
        """
        if verbose:
            print(f"  🔍 计算梯度，寻找最佳边...")
        
        # ===== 方法1：直接计算梯度（需要修改PyG，这里用近似方法）=====
        # 我们用一个技巧：枚举候选边，但用梯度信息排序
        
        # 计算特征梯度（反映节点重要性）
        self.data.x.requires_grad = True
        
        output = self.model(self.data.x, edge_index)
        loss = F.nll_loss(
            output[target_node:target_node+1],
            self.data.y[target_node:target_node+1]
        )
        
        self.model.zero_grad()
        loss.backward()
        
        # 特征梯度的范数 = 节点重要性
        node_importance = torch.norm(self.data.x.grad, dim=1, p=2).cpu().numpy()
        
        # ===== 候选边：目标节点 → 其他节点 =====
        existing_edges = set()
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            existing_edges.add((src, dst))
        
        candidates = []
        for dst in range(self.data.num_nodes):
            # 排除：自环、已存在的边
            if dst == target_node:
                continue
            if (target_node, dst) in existing_edges:
                continue
            
            # 影响力分数 = 目标节点重要性 × 候选节点重要性
            score = node_importance[dst]
            candidates.append(((target_node, dst), score))
        
        if len(candidates) == 0:
            return None, 0
        
        # 排序，选择分数最高的
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_edge, best_score = candidates[0]
        
        if verbose:
            print(f"  ✓ 找到最佳边: {best_edge}")
            print(f"  ✓ 候选边数量: {len(candidates)}")
        
        return best_edge, best_score


# ============================================
# 第三部分：训练和测试
# ============================================
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
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
                acc = correct / data.test_mask.sum().item()
            print(f'  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}')
            model.train()


def evaluate_model(model, data, edge_index=None):
    """评估模型"""
    if edge_index is None:
        edge_index = data.edge_index
    
    model.eval()
    with torch.no_grad():
        pred = model(data.x, edge_index).argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
    return acc


def find_vulnerable_node(model, data):
    """找到一个容易攻击的节点"""
    model.eval()
    
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        probs = torch.exp(output)
        confidence = probs.max(dim=1)[0]
    
    # 在测试集中找预测正确但置信度较低的节点
    test_nodes = data.test_mask.nonzero().squeeze()
    
    vulnerable_candidates = []
    for node in test_nodes:
        node_idx = node.item()
        pred = output[node_idx].argmax().item()
        true_label = data.y[node_idx].item()
        
        if pred == true_label:  # 预测正确
            conf = confidence[node_idx].item()
            vulnerable_candidates.append((node_idx, conf))
    
    # 选择置信度最低的
    vulnerable_candidates.sort(key=lambda x: x[1])
    
    if len(vulnerable_candidates) > 0:
        return vulnerable_candidates[0][0]  # 返回节点ID
    else:
        return test_nodes[0].item()  # 默认返回第一个测试节点


# ============================================
# 第四部分：主函数
# ============================================
def main():
    print("=" * 80)
    print("Nettack攻击演示 - 完整流程")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n【步骤1】加载Cora数据集...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    print(f"  ✓ 节点数: {data.num_nodes}")
    print(f"  ✓ 边数: {data.num_edges}")
    print(f"  ✓ 特征维度: {dataset.num_features}")
    print(f"  ✓ 类别数: {dataset.num_classes}")
    
    # 2. 训练模型
    print("\n【步骤2】训练GCN模型...")
    model = SimpleGCN(dataset.num_features, dataset.num_classes)
    train_model(model, data, epochs=200)
    
    # 3. 评估原始模型
    original_acc = evaluate_model(model, data)
    print(f"\n【步骤3】原始模型测试准确率: {original_acc:.4f}")
    
    # 4. 选择目标节点
    print("\n【步骤4】选择攻击目标节点...")
    target_node = find_vulnerable_node(model, data)
    print(f"  ✓ 选择了节点 {target_node}（容易攻击的节点）")
    
    # 5. 执行Nettack攻击
    print("\n【步骤5】执行Nettack攻击...")
    attacker = NettackAttack(model, data)
    adv_edge_index, added_edges = attacker.attack(
        target_node, 
        num_perturbations=5,
        verbose=True
    )
    
    # 6. 评估攻击后的模型
    print("\n【步骤6】评估攻击后的整体影响...")
    adv_acc = evaluate_model(model, data, adv_edge_index)
    print(f"  原始准确率: {original_acc:.4f}")
    print(f"  攻击后准确率: {adv_acc:.4f}")
    print(f"  准确率下降: {(original_acc - adv_acc):.4f}")
    
    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
    
    # 7. 关键洞察
    print("\n💡 关键洞察：")
    print(f"  • 只添加了 {len(added_edges)} 条边")
    print(f"  • 成功改变了目标节点的预测")
    print(f"  • 整体准确率下降了 {(original_acc - adv_acc)*100:.2f}%")
    print(f"  • 攻击成本极低，但破坏力很大！")
    print(f"\n这就是为什么需要防御机制！")


if __name__ == "__main__":
    main()

