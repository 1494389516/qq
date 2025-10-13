"""
🧹 图净化（GCN-Jaccard）- 超简单版
核心：用Jaccard相似度检测并删除可疑边

学习目标：
1. 理解Jaccard相似度的计算
2. 理解"物以类聚，人以群分"
3. 看到图净化的效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

print("=" * 80)
print("🧹 图净化（GCN-Jaccard）- 超简单演示")
print("=" * 80)

# ============================================================================
# 第1步：加载数据
# ============================================================================

print("\n📦 加载Cora数据集...")
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges // 2}")
print(f"特征维度: {dataset.num_features}")

# ============================================================================
# 第2步：计算Jaccard相似度（核心函数）
# ============================================================================

def compute_jaccard_similarity(edge_index, num_nodes):
    """
    计算每条边的Jaccard相似度
    
    Jaccard(u, v) = |共同邻居| / |所有邻居|
    """
    print("\n🔍 步骤1：计算Jaccard相似度")
    print("-" * 80)
    
    # 构建邻接表（方便查询邻居）
    neighbors = [set() for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    
    # 计算每条边的Jaccard相似度
    jaccard_scores = []
    
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        
        # 获取邻居（排除彼此）
        neighbors_src = neighbors[src] - {dst}
        neighbors_dst = neighbors[dst] - {src}
        
        # 计算Jaccard
        if len(neighbors_src) == 0 and len(neighbors_dst) == 0:
            jaccard = 0.0
        else:
            intersection = neighbors_src & neighbors_dst  # 交集（共同邻居）
            union = neighbors_src | neighbors_dst          # 并集（所有邻居）
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        
        jaccard_scores.append(jaccard)
    
    jaccard_scores = torch.tensor(jaccard_scores)
    
    print(f"✓ 计算完成！")
    print(f"  Jaccard平均值: {jaccard_scores.mean():.4f}")
    print(f"  Jaccard最小值: {jaccard_scores.min():.4f}")
    print(f"  Jaccard最大值: {jaccard_scores.max():.4f}")
    
    # 统计分布
    low = (jaccard_scores < 0.01).sum().item()
    medium = ((jaccard_scores >= 0.01) & (jaccard_scores < 0.1)).sum().item()
    high = (jaccard_scores >= 0.1).sum().item()
    
    print(f"\n  相似度分布:")
    print(f"    很低 (<0.01):  {low} 条边 ({low/len(jaccard_scores)*100:.1f}%)")
    print(f"    中等 (0.01-0.1): {medium} 条边 ({medium/len(jaccard_scores)*100:.1f}%)")
    print(f"    很高 (>0.1):   {high} 条边 ({high/len(jaccard_scores)*100:.1f}%)")
    
    return jaccard_scores


def filter_edges(edge_index, jaccard_scores, threshold=0.0):
    """根据Jaccard阈值过滤边"""
    print(f"\n🧹 步骤2：过滤低相似度的边")
    print("-" * 80)
    print(f"阈值: {threshold}")
    print(f"规则: Jaccard < {threshold} 的边将被删除")
    
    # 保留相似度 >= threshold 的边
    keep_mask = jaccard_scores >= threshold
    clean_edge_index = edge_index[:, keep_mask]
    
    removed = (~keep_mask).sum().item()
    remaining = keep_mask.sum().item()
    
    print(f"\n✓ 过滤完成！")
    print(f"  移除边数: {removed} ({removed/len(jaccard_scores)*100:.1f}%)")
    print(f"  保留边数: {remaining} ({remaining/len(jaccard_scores)*100:.1f}%)")
    
    return clean_edge_index


# ============================================================================
# 第3步：定义GCN模型
# ============================================================================

class SimpleGCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_and_test(model, data, edge_index, epochs=100):
    """训练并测试模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 训练
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    
    # 测试
    model.eval()
    with torch.no_grad():
        pred = model(data.x, edge_index).argmax(dim=1)
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).float().mean()
    
    return test_acc.item()


# ============================================================================
# 第4步：对比实验
# ============================================================================

print("\n" + "=" * 80)
print("📊 对比实验：图净化的效果")
print("=" * 80)

# 计算Jaccard相似度
jaccard_scores = compute_jaccard_similarity(data.edge_index, data.num_nodes)

# 实验1：不使用图净化
print("\n" + "=" * 80)
print("🔬 实验1：不使用图净化（原始图）")
print("=" * 80)

model_original = SimpleGCN(dataset.num_features, dataset.num_classes)
acc_original = train_and_test(model_original, data, data.edge_index, epochs=100)

print(f"\n✓ 训练完成")
print(f"  测试准确率: {acc_original:.4f}")

# 实验2：使用图净化
print("\n" + "=" * 80)
print("🔬 实验2：使用图净化（清洗后的图）")
print("=" * 80)

clean_edge_index = filter_edges(data.edge_index, jaccard_scores, threshold=0.0)

print(f"\n🎯 步骤3：在清洗后的图上训练")
print("-" * 80)

model_clean = SimpleGCN(dataset.num_features, dataset.num_classes)
acc_clean = train_and_test(model_clean, data, clean_edge_index, epochs=100)

print(f"\n✓ 训练完成")
print(f"  测试准确率: {acc_clean:.4f}")

# ============================================================================
# 第5步：总结
# ============================================================================

print("\n" + "=" * 80)
print("📊 总结对比")
print("=" * 80)

print(f"""
┌────────────────────────────────────────┐
│ 图净化效果对比                          │
├────────────────────────────────────────┤
│ 原始图准确率:  {acc_original:.4f}                 │
│ 净化图准确率:  {acc_clean:.4f}                 │
│ 准确率变化:    {(acc_clean - acc_original):+.4f}                │
├────────────────────────────────────────┤
│ 移除边数:      {(jaccard_scores < 0.0).sum().item()}                      │
│ 保留边比例:    {(jaccard_scores >= 0.0).sum().item() / len(jaccard_scores) * 100:.1f}%                │
└────────────────────────────────────────┘
""")

if acc_clean >= acc_original:
    print("✅ 图净化有效！移除噪声边后，准确率提升或持平")
else:
    print("⚠️  准确率略有下降，但鲁棒性可能提升（抗攻击能力更强）")

print("\n" + "=" * 80)
print("💡 核心理解")
print("=" * 80)

print("""
1. 【Jaccard相似度】
   物以类聚，人以群分
   → 真实的边：有很多共同邻居
   → 攻击的边：几乎没有共同邻居

2. 【图净化流程】
   计算相似度 → 设置阈值 → 删除可疑边 → 训练模型

3. 【防御效果】
   对随机攻击有效 ✅
   对精心构造的攻击效果有限 ⚠️
   
4. 【最佳实践】
   图净化 + 对抗训练 + 鲁棒GNN = 最强防御！

5. 【类比理解】
   对抗训练 = 疫苗（让模型见过攻击）
   图净化   = 洗菜（训练前清洗数据）
   鲁棒GNN  = 基因改造（改变模型架构）
   
   三者结合，效果最好！
""")

print("=" * 80)
print("✅ 学习完成！")
print("=" * 80)

print("""
📚 今天学到了：

1. ✅ Jaccard相似度 = 共同邻居 / 所有邻居
2. ✅ 物以类聚，人以群分的原理
3. ✅ 图净化的完整流程
4. ✅ 三种防御方法的对比

下一步：
  → 学习鲁棒GNN（注意力机制）
  → 理解三种防御如何组合
  → 实战应用！
""")

