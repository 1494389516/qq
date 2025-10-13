"""
Metattack: 基于元学习的图对抗攻击
论文：Adversarial Attacks on Graph Neural Networks via Meta Learning (ICLR 2019)
作者：Daniel Zügner, Stephan Günnemann

核心思想：
- 训练时投毒攻击（Poisoning Attack）
- 用元学习快速近似重训练
- 全局攻击（影响整个模型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
import copy
from tqdm import tqdm

print("=" * 80)
print("🎯 Metattack: 训练时投毒攻击")
print("=" * 80)


# ============================================================================
# 第一部分：GCN模型定义
# ============================================================================

class GCN(nn.Module):
    """图卷积网络"""
    
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
# 第二部分：Metattack核心算法
# ============================================================================

class Metattack:
    """
    Metattack攻击算法
    
    核心步骤：
    1. 初始化：干净的图 G = (A, X)
    2. 生成候选边：哪些边可以修改
    3. 元学习梯度：快速估算修改后的影响
    4. 贪心选择：选择影响最大的修改
    5. 重复直到达到预算
    """
    
    def __init__(self, model, data, budget=20, lr=0.01):
        """
        参数：
        - model: 目标GNN模型
        - data: 图数据
        - budget: 扰动预算（最多修改多少条边）
        - lr: 元学习的学习率
        """
        self.model = model
        self.data = data
        self.budget = budget
        self.lr = lr
        
        # 保存原始邻接矩阵
        self.ori_edge_index = data.edge_index.clone()
        
    def attack(self):
        """执行Metattack"""
        
        print("\n" + "=" * 80)
        print("开始Metattack攻击")
        print("=" * 80)
        print(f"扰动预算: {self.budget}条边")
        print(f"原始边数: {self.data.edge_index.shape[1] // 2}条")
        
        # 当前的边索引
        modified_edge_index = self.ori_edge_index.clone()
        
        # 记录每次修改
        modifications = []
        
        for step in tqdm(range(self.budget), desc="攻击进度"):
            # 1. 生成候选修改
            candidates = self._generate_candidates(modified_edge_index)
            
            if len(candidates) == 0:
                print(f"\n警告：第{step+1}步没有候选边了")
                break
            
            # 2. 评估每个候选的攻击效果
            best_candidate = None
            best_score = -float('inf')
            
            # 随机采样候选边（全部评估太慢）
            sample_size = min(100, len(candidates))
            sampled_candidates = np.random.choice(
                len(candidates), 
                size=sample_size, 
                replace=False
            )
            
            for idx in sampled_candidates:
                candidate = candidates[idx]
                
                # 尝试这个修改
                edge_index_try = self._apply_modification(
                    modified_edge_index, 
                    candidate
                )
                
                # 用元学习评估效果
                score = self._meta_gradient_attack(edge_index_try)
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            # 3. 执行最佳修改
            if best_candidate is not None:
                modified_edge_index = self._apply_modification(
                    modified_edge_index, 
                    best_candidate
                )
                modifications.append(best_candidate)
                
                if (step + 1) % 5 == 0:
                    print(f"\n第{step+1}步: 修改边 {best_candidate}, 攻击得分={best_score:.4f}")
        
        print("\n" + "=" * 80)
        print("攻击完成！")
        print("=" * 80)
        print(f"总共修改了 {len(modifications)} 条边")
        
        return modified_edge_index, modifications
    
    def _generate_candidates(self, edge_index):
        """
        生成候选边
        
        策略：
        1. 可以删除现有的边
        2. 可以添加不存在的边
        3. 优先考虑度数低的节点（容易影响）
        """
        num_nodes = self.data.num_nodes
        
        # 将edge_index转为集合（方便查询）
        existing_edges = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u < v:  # 只存储一个方向
                existing_edges.add((u, v))
        
        candidates = []
        
        # 策略1：删除边（从现有边中选）
        for edge in list(existing_edges)[:100]:  # 限制数量
            candidates.append(('delete', edge[0], edge[1]))
        
        # 策略2：添加边（随机采样）
        for _ in range(100):
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v and (min(u,v), max(u,v)) not in existing_edges:
                candidates.append(('add', u, v))
        
        return candidates
    
    def _apply_modification(self, edge_index, modification):
        """应用一个修改（添加或删除边）"""
        
        action, u, v = modification
        
        if action == 'add':
            # 添加边 (u,v) 和 (v,u)
            new_edges = torch.tensor([[u, v], [v, u]], dtype=torch.long).T
            edge_index_new = torch.cat([edge_index, new_edges], dim=1)
            
        else:  # delete
            # 删除边 (u,v) 和 (v,u)
            mask = ~((edge_index[0] == u) & (edge_index[1] == v) |
                     (edge_index[0] == v) & (edge_index[1] == u))
            edge_index_new = edge_index[:, mask]
        
        return edge_index_new
    
    def _meta_gradient_attack(self, edge_index_modified):
        """
        核心：元学习梯度近似
        
        思想：
        不真的重新训练模型（太慢）
        用一步梯度近似重训练后的效果
        
        公式：
        θ* ≈ θ - α * ∇_θ L_train(θ, A')
        
        然后评估测试集损失：
        L_test(θ*, A')
        
        损失越大 = 攻击越成功
        """
        
        # 1. 在修改后的图上计算训练损失的梯度
        self.model.train()
        self.model.zero_grad()
        
        out = self.model(self.data.x, edge_index_modified)
        loss_train = F.nll_loss(
            out[self.data.train_mask],
            self.data.y[self.data.train_mask]
        )
        
        loss_train.backward()
        
        # 2. 保存梯度（用于元学习）
        meta_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                meta_grads.append(param.grad.clone())
            else:
                meta_grads.append(torch.zeros_like(param))
        
        # 3. 用元梯度更新参数（模拟重训练）
        with torch.no_grad():
            original_params = []
            for param in self.model.parameters():
                original_params.append(param.clone())
            
            # 一步梯度下降（近似重训练）
            for param, grad in zip(self.model.parameters(), meta_grads):
                param.data = param.data - self.lr * grad
            
            # 4. 在测试集上评估
            self.model.eval()
            out = self.model(self.data.x, edge_index_modified)
            loss_test = F.nll_loss(
                out[self.data.test_mask],
                self.data.y[self.data.test_mask]
            )
            
            attack_score = loss_test.item()
            
            # 5. 恢复参数
            for param, ori_param in zip(self.model.parameters(), original_params):
                param.data = ori_param
        
        return attack_score


# ============================================================================
# 第三部分：完整演示
# ============================================================================

def demonstrate_metattack():
    """完整演示Metattack的效果"""
    
    print("\n📦 加载数据集...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    print(f"数据集: Cora")
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges // 2}")
    print(f"特征维度: {dataset.num_features}")
    print(f"类别数: {dataset.num_classes}")
    
    # ========== 步骤1：训练干净的模型 ==========
    print("\n" + "=" * 80)
    print("步骤1：在干净的图上训练GCN")
    print("=" * 80)
    
    clean_model = GCN(dataset.num_features, dataset.num_classes)
    optimizer = torch.optim.Adam(clean_model.parameters(), lr=0.01, weight_decay=5e-4)
    
    clean_model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = clean_model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            clean_model.eval()
            with torch.no_grad():
                pred = clean_model(data.x, data.edge_index).argmax(dim=1)
                train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).float().mean()
                test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).float().mean()
            print(f"Epoch {epoch+1}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")
            clean_model.train()
    
    # 最终评估
    clean_model.eval()
    with torch.no_grad():
        pred = clean_model(data.x, data.edge_index).argmax(dim=1)
        clean_test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).float().mean().item()
    
    print(f"\n✓ 干净模型训练完成")
    print(f"✓ 测试准确率: {clean_test_acc:.4f}")
    
    # ========== 步骤2：执行Metattack ==========
    print("\n" + "=" * 80)
    print("步骤2：执行Metattack攻击（污染图结构）")
    print("=" * 80)
    
    attacker = Metattack(
        model=clean_model,
        data=data,
        budget=20,  # 修改20条边
        lr=0.01
    )
    
    poisoned_edge_index, modifications = attacker.attack()
    
    # ========== 步骤3：在被污染的图上重新训练 ==========
    print("\n" + "=" * 80)
    print("步骤3：在被污染的图上重新训练GCN")
    print("=" * 80)
    
    poisoned_model = GCN(dataset.num_features, dataset.num_classes)
    optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 创建被污染的数据
    poisoned_data = copy.deepcopy(data)
    poisoned_data.edge_index = poisoned_edge_index
    
    poisoned_model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = poisoned_model(poisoned_data.x, poisoned_data.edge_index)
        loss = F.nll_loss(out[poisoned_data.train_mask], poisoned_data.y[poisoned_data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            poisoned_model.eval()
            with torch.no_grad():
                pred = poisoned_model(poisoned_data.x, poisoned_data.edge_index).argmax(dim=1)
                train_acc = pred[poisoned_data.train_mask].eq(poisoned_data.y[poisoned_data.train_mask]).float().mean()
                test_acc = pred[poisoned_data.test_mask].eq(poisoned_data.y[poisoned_data.test_mask]).float().mean()
            print(f"Epoch {epoch+1}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")
            poisoned_model.train()
    
    # 最终评估
    poisoned_model.eval()
    with torch.no_grad():
        pred = poisoned_model(poisoned_data.x, poisoned_data.edge_index).argmax(dim=1)
        poisoned_test_acc = pred[poisoned_data.test_mask].eq(poisoned_data.y[poisoned_data.test_mask]).float().mean().item()
    
    print(f"\n✓ 被污染模型训练完成")
    print(f"✓ 测试准确率: {poisoned_test_acc:.4f}")
    
    # ========== 步骤4：对比结果 ==========
    print("\n" + "=" * 80)
    print("📊 攻击效果对比")
    print("=" * 80)
    
    acc_drop = clean_test_acc - poisoned_test_acc
    drop_percentage = (acc_drop / clean_test_acc) * 100
    
    print(f"\n干净图上的准确率: {clean_test_acc:.4f}")
    print(f"被污染图上的准确率: {poisoned_test_acc:.4f}")
    print(f"准确率下降: {acc_drop:.4f} ({drop_percentage:.1f}%)")
    
    print(f"\n扰动统计:")
    print(f"  修改边数: {len(modifications)}")
    print(f"  原始边数: {data.num_edges // 2}")
    print(f"  扰动比例: {len(modifications) / (data.num_edges // 2) * 100:.2f}%")
    
    # 分析修改类型
    add_count = sum(1 for m in modifications if m[0] == 'add')
    delete_count = sum(1 for m in modifications if m[0] == 'delete')
    
    print(f"\n修改类型:")
    print(f"  添加边: {add_count}")
    print(f"  删除边: {delete_count}")
    
    print("\n" + "=" * 80)
    print("💡 Metattack的关键特点")
    print("=" * 80)
    print("""
1. 训练时投毒（Poisoning Attack）：
   - 在训练阶段就污染数据
   - 比测试时攻击（Nettack）更隐蔽
   
2. 全局攻击：
   - 影响整个模型，不只是某个节点
   - 所有节点的分类性能都下降
   
3. 元学习加速：
   - 不需要真的重新训练模型
   - 用一步梯度近似重训练效果
   - 速度快100倍！
   
4. 实战意义：
   - 更接近真实的黑产攻击
   - 黑产会在数据收集阶段就开始污染
   - 防御更困难
    """)
    
    return clean_test_acc, poisoned_test_acc


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("🎓 Metattack算法学习")
    print("=" * 80)
    
    print("""
Metattack vs Nettack 对比：

┌────────────────────────────────────────┐
│ Nettack (2018)                         │
├────────────────────────────────────────┤
│ • 测试时攻击（Evasion）                │
│ • 目标攻击（单个节点）                 │
│ • 局部影响                             │
│ • 容易被发现                           │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ Metattack (2019)                       │
├────────────────────────────────────────┤
│ • 训练时投毒（Poisoning）              │
│ • 全局攻击（整个模型）                 │
│ • 广泛影响                             │
│ • 更隐蔽，更难防御                     │
└────────────────────────────────────────┘
    """)
    
    # 运行完整演示
    clean_acc, poisoned_acc = demonstrate_metattack()
    
    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
    
    print(f"""
关键数据：
- 攻击前准确率: {clean_acc:.1%}
- 攻击后准确率: {poisoned_acc:.1%}
- 性能下降: {(clean_acc - poisoned_acc):.1%}

这就是为什么Metattack如此危险：
只修改很少的边（<1%），就能让模型性能大幅下降！
    """)

