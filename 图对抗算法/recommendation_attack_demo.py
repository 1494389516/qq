"""
推荐系统刷单攻击可视化演示
让您"看到"黑产是如何攻击推荐系统的！
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🛍️  推荐系统刷单攻击演示")
print("=" * 80)

# ===== 场景设置 =====
print("\n📊 场景：简化的电商推荐系统")
print("-" * 80)

# 10个用户，10个商品
num_users = 10
num_items = 10
num_total = num_users + num_items

print(f"  用户数：{num_users}（用户1-10）")
print(f"  商品数：{num_items}（商品A-J）")

# ===== 构建初始图 =====
print("\n🔧 构建初始用户-商品关系图...")

# 正常的购买关系（稀疏的）
# 用二部图表示：用户 -- 商品
purchases = [
    (0, 10),  # 用户1买了商品A
    (0, 11),  # 用户1买了商品B
    (1, 10),  # 用户2买了商品A
    (1, 12),  # 用户2买了商品C
    (2, 11),  # 用户3买了商品B
    (2, 12),  # 用户3买了商品C
    (3, 13),  # 用户4买了商品D
    (4, 13),  # 用户5买了商品D
    (5, 14),  # 用户6买了商品E
    (6, 15),  # 用户7买了商品F
    (7, 15),  # 用户8买了商品F
    (8, 16),  # 用户9买了商品G
    (9, 17),  # 用户10买了商品H
]

print(f"✓ 初始购买关系：{len(purchases)}条")
print("\n购买明细：")
for u, i in purchases[:5]:
    print(f"  用户{u+1} 购买了 商品{chr(65+i-10)}")
print("  ...")

# 计算每个商品的初始热度（购买次数）
item_popularity = {i: 0 for i in range(10, 20)}
for u, i in purchases:
    item_popularity[i] += 1

print(f"\n📈 商品初始热度排行：")
sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
for item_id, count in sorted_items[:5]:
    item_name = chr(65 + item_id - 10)
    print(f"  商品{item_name}: {count}次购买 {'🔥' if count >= 2 else ''}")

# ===== 目标商品 =====
target_item = 18  # 商品I（目前0购买）
target_item_name = chr(65 + target_item - 10)
print(f"\n🎯 目标商品：商品{target_item_name}")
print(f"   初始购买数：{item_popularity[target_item]}次")
print(f"   初始排名：{sorted(item_popularity.values(), reverse=True).index(item_popularity[target_item]) + 1}/10")
print(f"   → 黑产想要刷这个商品，让它进入推荐列表！")

# ===== 模拟简单的推荐算法 =====
def recommend(purchases, user_id, top_k=3):
    """
    简单的协同过滤推荐
    逻辑：推荐热门商品（购买次数多的）
    """
    # 计算商品热度
    item_counts = {}
    for u, i in purchases:
        item_counts[i] = item_counts.get(i, 0) + 1
    
    # 已购买的商品不推荐
    bought_items = set([i for u, i in purchases if u == user_id])
    
    # 推荐热门且未购买的商品
    candidates = [(i, count) for i, count in item_counts.items() 
                  if i not in bought_items]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    return [i for i, _ in candidates[:top_k]]

# ===== 攻击前的推荐结果 =====
print("\n" + "=" * 80)
print("📱 攻击前：推荐系统的表现")
print("=" * 80)

test_user = 0  # 测试用户1
before_rec = recommend(purchases, test_user)
print(f"\n给【用户1】的推荐列表（Top 3）：")
for rank, item_id in enumerate(before_rec, 1):
    item_name = chr(65 + item_id - 10)
    pop = item_popularity[item_id]
    print(f"  {rank}. 商品{item_name} (热度:{pop})")

target_in_rec = target_item in before_rec
print(f"\n❓ 目标商品{target_item_name}在推荐中吗？ {'✓ 是' if target_in_rec else '✗ 否'}")

# ===== 刷单攻击！=====
print("\n" + "=" * 80)
print("⚔️  刷单攻击开始！")
print("=" * 80)

print(f"\n黑产策略：")
print(f"  1. 注册5个小号（虚假账号）")
print(f"  2. 每个小号购买目标商品{target_item_name}")
print(f"  3. 制造'热销'假象")

# 创建刷单账号和购买记录
fake_users = list(range(10, 15))  # 小号：用户11-15
fake_purchases = [(u, target_item) for u in fake_users]

print(f"\n执行刷单中...")
for i, (u, item) in enumerate(fake_purchases, 1):
    print(f"  [{i}/5] 小号{u-9}购买了商品{target_item_name}")

# 攻击后的购买关系
purchases_after = purchases + fake_purchases

# 更新热度
item_popularity_after = {i: 0 for i in range(10, 20)}
for u, i in purchases_after:
    item_popularity_after[i] += 1

print(f"\n✓ 刷单完成！")
print(f"\n📊 商品{target_item_name}的变化：")
print(f"  购买次数：{item_popularity[target_item]} → {item_popularity_after[target_item]} (+{item_popularity_after[target_item] - item_popularity[target_item]})")
before_rank = sorted(item_popularity.values(), reverse=True).index(item_popularity[target_item]) + 1
after_rank = sorted(item_popularity_after.values(), reverse=True).index(item_popularity_after[target_item]) + 1
print(f"  热度排名：{before_rank}/10 → {after_rank}/10 (↑{before_rank - after_rank}名)")

# ===== 攻击后的推荐结果 =====
print("\n" + "=" * 80)
print("📱 攻击后：推荐系统的表现")
print("=" * 80)

after_rec = recommend(purchases_after, test_user)
print(f"\n给【用户1】的推荐列表（Top 3）：")
for rank, item_id in enumerate(after_rec, 1):
    item_name = chr(65 + item_id - 10)
    pop = item_popularity_after[item_id]
    is_target = "← 🎯目标商品！" if item_id == target_item else ""
    print(f"  {rank}. 商品{item_name} (热度:{pop}) {is_target}")

target_in_rec_after = target_item in after_rec
print(f"\n❓ 目标商品{target_item_name}在推荐中吗？ {'✓ 是' if target_in_rec_after else '✗ 否'}")

# ===== 攻击效果对比 =====
print("\n" + "=" * 80)
print("📊 攻击效果对比")
print("=" * 80)

print(f"\n{'指标':<20} {'攻击前':<15} {'攻击后':<15} {'变化'}")
print("-" * 70)
print(f"{'目标商品购买数':<20} {item_popularity[target_item]:<15} {item_popularity_after[target_item]:<15} +{item_popularity_after[target_item] - item_popularity[target_item]}")
print(f"{'目标商品排名':<20} {before_rank:<15} {after_rank:<15} ↑{before_rank - after_rank}名")
print(f"{'是否进入推荐':<20} {'否':<15} {'是' if target_in_rec_after else '否':<15} {'✓ 成功' if target_in_rec_after and not target_in_rec else '失败'}")

# ===== 可视化 =====
print("\n" + "=" * 80)
print("📊 生成可视化图表...")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1：商品热度对比
ax1 = axes[0]
items = [chr(65 + i - 10) for i in range(10, 20)]
before_counts = [item_popularity.get(i, 0) for i in range(10, 20)]
after_counts = [item_popularity_after.get(i, 0) for i in range(10, 20)]

x = np.arange(len(items))
width = 0.35

bars1 = ax1.bar(x - width/2, before_counts, width, label='攻击前', alpha=0.8, color='skyblue')
bars2 = ax1.bar(x + width/2, after_counts, width, label='攻击后', alpha=0.8, color='salmon')

# 标记目标商品
target_idx = target_item - 10
ax1.bar(target_idx + width/2, after_counts[target_idx], width, 
        color='red', alpha=0.8, label='目标商品')

ax1.set_xlabel('商品', fontsize=12)
ax1.set_ylabel('购买次数', fontsize=12)
ax1.set_title('刷单攻击前后：商品热度对比', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(items)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 图2：推荐列表变化
ax2 = axes[1]

before_rec_names = [chr(65 + i - 10) for i in before_rec]
after_rec_names = [chr(65 + i - 10) for i in after_rec]

y_pos = np.arange(3)
ax2.barh(y_pos + 0.2, [3, 2, 1], 0.35, label='攻击前', color='skyblue', alpha=0.8)
ax2.barh(y_pos - 0.2, [3, 2, 1], 0.35, label='攻击后', color='salmon', alpha=0.8)

# 标注商品名称
for i, (before, after) in enumerate(zip(before_rec_names, after_rec_names)):
    ax2.text(3.2, i + 0.2, before, va='center', fontsize=10)
    ax2.text(3.2, i - 0.2, after, va='center', fontsize=10, 
            weight='bold' if after == target_item_name else 'normal',
            color='red' if after == target_item_name else 'black')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(['Top 1', 'Top 2', 'Top 3'])
ax2.set_xlabel('推荐优先级', fontsize=12)
ax2.set_title('推荐列表变化（用户1）', fontsize=14, fontweight='bold')
ax2.legend()
ax2.invert_yaxis()
ax2.set_xlim([0, 4])

plt.tight_layout()
plt.savefig('/Users/mac/Desktop/对抗算法/recommendation_attack.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存：recommendation_attack.png")

# ===== 总结 =====
print("\n" + "=" * 80)
print("💡 关键洞察")
print("=" * 80)

print("""
【攻击成本】
- 注册5个小号
- 购买5次（假设每次50元）
- 总成本：250元

【攻击收益】
- 商品进入推荐列表
- 曝光量增加10倍+
- 如果转化率5%，销量增加100+
- 潜在收益：几千到几万元

【ROI】
投入250元 → 收益几千元 → ROI: 10-50倍

→ 这就是为什么黑产要刷单！
→ 这就是为什么需要防御！

【真实世界】
本demo只有10个用户、10个商品
真实场景：
- 百万用户
- 千万商品
- 刷单更隐蔽
- 影响更大

【防御思路】
1. 检测异常增长（商品I突然5个购买）
2. 检测账号可疑（新注册立即购买）
3. 降低可疑购买的权重
4. 人工审核热度突增的商品
""")

print("\n" + "=" * 80)
print("✅ 演示完成！")
print("=" * 80)
print("\n现在您能'看到'黑产是如何攻击推荐系统的了！")
print("📊 查看图表：recommendation_attack.png")
