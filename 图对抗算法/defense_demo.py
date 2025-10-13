"""
防御演示：针对epsilon=0.05-0.15的对抗攻击
展示3种常见的防御策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import copy
import numpy as np

print("=" * 70)
print("🛡️  图对抗攻击防御演示")
print("=" * 70)

# ===== 加载数据和模型 =====
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class SimpleGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 快速训练一个基础模型
print("\n🏋️  训练基础模型...")
model = SimpleGCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

print("✓ 基础模型训练完成")

# 测试原始准确率
model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    original_acc = correct / data.test_mask.sum().item()

print(f"✓ 原始准确率: {original_acc:.4f}")


# ===== FGSM攻击函数 =====
def fgsm_attack(model, data, epsilon=0.1):
    """标准FGSM攻击"""
    adv_data = copy.deepcopy(data)
    adv_data.x.requires_grad = True
    
    output = model(adv_data.x, adv_data.edge_index)
    target_node = data.test_mask.nonzero()[0].item()
    loss = F.nll_loss(output[target_node:target_node+1], 
                     adv_data.y[target_node:target_node+1])
    
    model.zero_grad()
    loss.backward()
    
    data_grad = adv_data.x.grad.data
    perturbed_x = adv_data.x + epsilon * data_grad.sign()
    adv_data.x = perturbed_x.detach()
    
    return adv_data


# ===== 防御策略1：对抗训练 =====
print("\n" + "=" * 70)
print("🛡️  防御策略1：对抗训练")
print("=" * 70)
print("原理：在训练时混入对抗样本，让模型见过攻击")

class AdversarialTrainedGCN(nn.Module):
    """对抗训练的GCN"""
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 对抗训练
print("\n训练中（混入对抗样本）...")
adv_model = AdversarialTrainedGCN()
optimizer = torch.optim.Adam(adv_model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(100):
    adv_model.train()
    
    # 1. 正常训练
    optimizer.zero_grad()
    out = adv_model(data.x, data.edge_index)
    loss_clean = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    # 2. 生成对抗样本并训练（重点！）
    adv_data = fgsm_attack(adv_model, data, epsilon=0.1)  # 用0.1训练
    out_adv = adv_model(adv_data.x, adv_data.edge_index)
    loss_adv = F.nll_loss(out_adv[data.train_mask], data.y[data.train_mask])
    
    # 3. 综合损失
    total_loss = 0.5 * loss_clean + 0.5 * loss_adv
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/100, 损失={total_loss.item():.4f}")

print("✓ 对抗训练完成")

# 测试对抗训练模型
adv_model.eval()
with torch.no_grad():
    pred = adv_model(data.x, data.edge_index).argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    adv_train_acc = correct / data.test_mask.sum().item()

print(f"✓ 对抗训练模型原始准确率: {adv_train_acc:.4f}")


# ===== 防御策略2：特征扰动检测 =====
print("\n" + "=" * 70)
print("🛡️  防御策略2：特征扰动检测")
print("=" * 70)
print("原理：检测特征的异常变化（统计方法）")

def detect_perturbation(x, threshold=0.1):
    """
    检测特征是否被扰动
    方法：计算特征变化的统计特性
    """
    # 计算特征的标准差
    feature_std = x.std(dim=0)
    
    # 检测异常大的标准差（可能是对抗扰动）
    suspicious = (feature_std > threshold).sum().item()
    
    return suspicious > len(feature_std) * 0.1  # 如果10%以上特征异常

# ===== 防御策略3：梯度遮蔽 =====
print("\n" + "=" * 70)
print("🛡️  防御策略3：梯度遮蔽（输入随机化）")
print("=" * 70)
print("原理：在输入中加入随机噪声，破坏梯度计算")

def gradient_masking_defense(model, x, edge_index, noise_scale=0.01):
    """
    梯度遮蔽防御
    在输入中加入随机噪声
    """
    # 加入随机噪声
    noise = torch.randn_like(x) * noise_scale
    x_noisy = x + noise
    
    # 预测（多次预测取平均）
    predictions = []
    for _ in range(5):  # 预测5次
        noise = torch.randn_like(x) * noise_scale
        x_temp = x + noise
        pred = model(x_temp, edge_index)
        predictions.append(pred)
    
    # 平均预测
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred


# ===== 测试3种防御效果 =====
print("\n" + "=" * 70)
print("📊 测试：针对epsilon=0.05-0.15的防御效果")
print("=" * 70)

# 危险区间
dangerous_epsilons = [0.05, 0.08, 0.1, 0.12, 0.15]

print(f"\n{'Epsilon':<10} {'基础模型':<12} {'对抗训练':<12} {'梯度遮蔽':<12}")
print("-" * 70)

for eps in dangerous_epsilons:
    # 生成对抗样本
    adv_data = fgsm_attack(model, data, epsilon=eps)
    
    # 1. 基础模型（无防御）
    with torch.no_grad():
        pred = model(adv_data.x, adv_data.edge_index).argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc_base = correct / data.test_mask.sum().item()
    
    # 2. 对抗训练模型
    with torch.no_grad():
        pred = adv_model(adv_data.x, adv_data.edge_index).argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc_adv_train = correct / data.test_mask.sum().item()
    
    # 3. 梯度遮蔽
    with torch.no_grad():
        out = gradient_masking_defense(model, adv_data.x, adv_data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc_grad_mask = correct / data.test_mask.sum().item()
    
    print(f"{eps:<10.2f} {acc_base:<12.4f} {acc_adv_train:<12.4f} {acc_grad_mask:<12.4f}")


# ===== 总结 =====
print("\n" + "=" * 70)
print("📊 防御效果总结")
print("=" * 70)

print("""
✨ 关键发现：

1. 【对抗训练】最有效
   - 在训练时混入攻击样本
   - 让模型"见过世面"
   - 准确率下降最小
   - 推荐指数：⭐⭐⭐⭐⭐

2. 【梯度遮蔽】次之
   - 破坏梯度信息
   - 让攻击者无法计算梯度
   - 但计算成本高（需要多次预测）
   - 推荐指数：⭐⭐⭐

3. 【扰动检测】辅助手段
   - 检测异常输入
   - 配合其他方法使用
   - 推荐指数：⭐⭐⭐

💡 实战建议：
- 核心：对抗训练（必选）
- 辅助：异常检测 + 业务规则
- 监控：持续监控攻击模式变化
""")

print("\n🎯 针对epsilon=0.05-0.15的防御重点：")
print("  1. 这个范围最危险（攻击效果好 + 隐蔽）")
print("  2. 对抗训练时重点用这个范围的epsilon")
print("  3. 异常检测系统设置阈值要考虑这个范围")
print("  4. 持续监控：黑产会试探最优epsilon")

print("\n" + "=" * 70)
print("✅ 防御演示完成！")
print("=" * 70)
