"""
经济防御算法演示 - 简化版
核心思想：让攻击"不划算"

场景：电商平台新人优惠券薅羊毛防御
目标：通过经济手段（提高成本+降低收益）让黑产放弃攻击

作者：学习者
日期：2025-10-12
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("💰 经济防御算法演示")
print("=" * 80)

# ============================================================================
# 第1部分：黑产的经济模型
# ============================================================================

class AttackerEconomics:
    """黑产的经济模型"""
    
    def __init__(self, name="默认"):
        self.name = name
        self.cost = {}
        self.revenue = {}
    
    def add_cost(self, item, amount):
        """添加成本项"""
        self.cost[item] = amount
    
    def add_revenue(self, item, amount):
        """添加收益项"""
        self.revenue[item] = amount
    
    def total_cost(self):
        """总成本"""
        return sum(self.cost.values())
    
    def total_revenue(self):
        """总收益"""
        return sum(self.revenue.values())
    
    def profit(self):
        """利润 = 收益 - 成本"""
        return self.total_revenue() - self.total_cost()
    
    def roi(self):
        """投资回报率 = 利润 / 成本"""
        cost = self.total_cost()
        if cost == 0:
            return 0
        return (self.profit() / cost) * 100
    
    def show_report(self):
        """显示报告"""
        print(f"\n{'='*60}")
        print(f"📊 经济分析报告：{self.name}")
        print(f"{'='*60}")
        
        print(f"\n【成本明细】")
        for item, amount in self.cost.items():
            print(f"  - {item:<20} {amount:>8.0f}元")
        print(f"  {'-'*40}")
        print(f"  总成本: {self.total_cost():>8.0f}元")
        
        print(f"\n【收益明细】")
        for item, amount in self.revenue.items():
            print(f"  - {item:<20} {amount:>8.0f}元")
        print(f"  {'-'*40}")
        print(f"  总收益: {self.total_revenue():>8.0f}元")
        
        profit = self.profit()
        roi = self.roi()
        
        print(f"\n【利润分析】")
        print(f"  利润: {profit:>8.0f}元")
        print(f"  ROI:  {roi:>8.1f}%")
        
        if profit > 0:
            print(f"\n  🔥 结论：有利可图！黑产会继续干！")
        elif profit == 0:
            print(f"\n  ⚖️  结论：不赚不赔，黑产可能放弃")
        else:
            print(f"\n  ✅ 结论：亏损{abs(profit):.0f}元，黑产放弃！")


# ============================================================================
# 第2部分：场景1 - 无防御状态
# ============================================================================

print("\n" + "=" * 80)
print("📍 场景1：无防御状态（黑产的天堂）")
print("=" * 80)

attacker_no_defense = AttackerEconomics("无防御状态")

# 黑产成本（很低）
attacker_no_defense.add_cost("手机号（接码平台）", 5)
attacker_no_defense.add_cost("IP代理", 10)
attacker_no_defense.add_cost("注册脚本", 20)
attacker_no_defense.add_cost("人力成本", 15)

# 黑产收益（很高）
attacker_no_defense.add_cost("养号消费（象征性）", 50)  # 买个便宜货凑单
attacker_no_defense.add_revenue("新人红包", 100)
attacker_no_defense.add_revenue("满减券", 200)
attacker_no_defense.add_revenue("首单券", 300)
attacker_no_defense.add_revenue("倒卖优惠券", 500)

attacker_no_defense.show_report()


# ============================================================================
# 第3部分：场景2 - 应用技术防御
# ============================================================================

print("\n\n" + "=" * 80)
print("📍 场景2：应用技术防御（治标不治本）")
print("=" * 80)

print("""
技术防御措施：
  1. 对抗训练 - 提升模型鲁棒性
  2. 异常检测 - 识别可疑账号
  3. 实时监控 - 快速发现攻击

效果：
  - 检测率从0% → 50%
  - 黑产成功率降低50%
  - 但黑产仍然有利可图！
""")

attacker_tech_defense = AttackerEconomics("技术防御")

# 成本略有提高（需要绕过检测）
attacker_tech_defense.add_cost("手机号", 5)
attacker_tech_defense.add_cost("高质量IP代理", 30)  # 提高
attacker_tech_defense.add_cost("反检测脚本", 50)    # 提高
attacker_tech_defense.add_cost("人力成本", 20)      # 提高
attacker_tech_defense.add_cost("养号消费", 50)

# 收益降低50%（因为检测率50%）
attacker_tech_defense.add_revenue("新人红包", 50)   # 减半
attacker_tech_defense.add_revenue("满减券", 100)     # 减半
attacker_tech_defense.add_revenue("首单券", 150)     # 减半
attacker_tech_defense.add_revenue("倒卖优惠券", 250) # 减半

attacker_tech_defense.show_report()


# ============================================================================
# 第4部分：场景3 - 应用经济防御（核心！）
# ============================================================================

print("\n\n" + "=" * 80)
print("📍 场景3：应用经济防御（釜底抽薪）")
print("=" * 80)

print("""
经济防御策略：

【第5层：提高成本】
  1. 强制实名认证（需要真实身份证）
     → 手机号成本：5元 → 200元（实名手机号）
  
  2. 必须真实消费满300元才能使用优惠券
     → 养号成本：50元 → 300元
  
  3. 设备指纹限制（一个设备只能注册1个账号）
     → 需要购买多个手机，成本×5
  
  4. 社交关系验证（需要3个实名好友）
     → 时间成本+3个月，人力成本×3

【第6层：降低收益】
  1. 新人红包额度：100元 → 20元
  
  2. 优惠券延迟到账：立即 → 30天后
     → 资金周转变慢，ROI降低
  
  3. 关联账户共享额度
     → 100个账户共享100元，而不是10000元
  
  4. 优惠券使用门槛：无门槛 → 满1000元可用
     → 倒卖价值大幅下降
""")

attacker_economic_defense = AttackerEconomics("经济防御")

# 成本大幅提高
attacker_economic_defense.add_cost("实名手机号", 200)        # 大幅提高
attacker_economic_defense.add_cost("高质量IP", 30)
attacker_economic_defense.add_cost("设备成本（手机）", 500)  # 新增
attacker_economic_defense.add_cost("真实消费要求", 300)      # 大幅提高
attacker_economic_defense.add_cost("人力成本（养号3个月）", 200)  # 大幅提高

# 收益大幅降低
attacker_economic_defense.add_revenue("新人红包（降额）", 20)      # 大幅降低
attacker_economic_defense.add_revenue("满减券（高门槛）", 30)       # 大幅降低
attacker_economic_defense.add_revenue("首单券（延迟到账）", 40)     # 大幅降低
attacker_economic_defense.add_revenue("倒卖券（价值暴跌）", 10)     # 大幅降低

attacker_economic_defense.show_report()


# ============================================================================
# 第5部分：场景4 - 最优防御（经济+技术+法律）
# ============================================================================

print("\n\n" + "=" * 80)
print("📍 场景4：最优防御组合（经济+技术+法律）")
print("=" * 80)

print("""
防御组合：

【技术层（1-4层）】
  - 数据清洗：移除异常账号
  - 对抗训练：提升模型鲁棒性
  - 动态重训练：每30天重训练
  - 实时监控：快速响应
  → 检测率：50% → 80%

【经济层（5-6层）】← 核心！
  - 提高成本：5倍
  - 降低收益：95%
  → 让攻击"不划算"

【法律层（第7层）】
  - 实名制可追溯
  - 配合警方打击
  - 公开判例威慑
  → 心理成本：无限大
""")

attacker_best_defense = AttackerEconomics("最优防御组合")

# 成本极高（经济手段）
attacker_best_defense.add_cost("实名手机号", 200)
attacker_best_defense.add_cost("设备成本", 500)
attacker_best_defense.add_cost("真实消费", 300)
attacker_best_defense.add_cost("养号时间成本", 200)
attacker_best_defense.add_cost("高级反检测工具", 100)  # 技术防御的影响

# 收益极低（经济+技术）
# 技术检测率80% → 只有20%能成功
attacker_best_defense.add_revenue("新人红包（仅20%成功）", 20 * 0.2)
attacker_best_defense.add_revenue("满减券", 30 * 0.2)
attacker_best_defense.add_revenue("首单券", 40 * 0.2)
attacker_best_defense.add_revenue("倒卖券", 10 * 0.2)

# 法律风险成本（隐性）
print("\n  ⚠️  额外风险：坐牢风险（10年）+ 罚金5000万")
print("      → 心理成本：无限大")

attacker_best_defense.show_report()


# ============================================================================
# 第6部分：对比可视化
# ============================================================================

print("\n\n" + "=" * 80)
print("📊 生成对比图表...")
print("=" * 80)

scenarios = ["无防御", "技术防御", "经济防御", "最优组合"]
attackers = [
    attacker_no_defense,
    attacker_tech_defense,
    attacker_economic_defense,
    attacker_best_defense
]

costs = [a.total_cost() for a in attackers]
revenues = [a.total_revenue() for a in attackers]
profits = [a.profit() for a in attackers]
rois = [a.roi() for a in attackers]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：成本对比
ax1 = axes[0, 0]
bars = ax1.bar(scenarios, costs, color=['red', 'orange', 'lightblue', 'green'], alpha=0.7)
ax1.set_ylabel('成本（元）', fontsize=12)
ax1.set_title('黑产成本对比', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, (bar, cost) in enumerate(zip(bars, costs)):
    ax1.text(i, cost + 20, f'{cost:.0f}元', ha='center', va='bottom', fontsize=10)

# 图2：收益对比
ax2 = axes[0, 1]
bars = ax2.bar(scenarios, revenues, color=['red', 'orange', 'lightblue', 'green'], alpha=0.7)
ax2.set_ylabel('收益（元）', fontsize=12)
ax2.set_title('黑产收益对比', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for i, (bar, rev) in enumerate(zip(bars, revenues)):
    ax2.text(i, rev + 20, f'{rev:.0f}元', ha='center', va='bottom', fontsize=10)

# 图3：利润对比
ax3 = axes[1, 0]
colors = ['red' if p > 0 else 'green' for p in profits]
bars = ax3.bar(scenarios, profits, color=colors, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.set_ylabel('利润（元）', fontsize=12)
ax3.set_title('黑产利润对比（核心指标！）', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, (bar, profit) in enumerate(zip(bars, profits)):
    va = 'bottom' if profit > 0 else 'top'
    offset = 20 if profit > 0 else -20
    ax3.text(i, profit + offset, f'{profit:.0f}元', ha='center', va=va, fontsize=10, fontweight='bold')

# 图4：ROI对比
ax4 = axes[1, 1]
colors = ['red' if r > 0 else 'green' for r in rois]
bars = ax4.bar(scenarios, rois, color=colors, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax4.set_ylabel('ROI（%）', fontsize=12)
ax4.set_title('投资回报率对比', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for i, (bar, roi) in enumerate(zip(bars, rois)):
    va = 'bottom' if roi > 0 else 'top'
    offset = 5 if roi > 0 else -5
    ax4.text(i, roi + offset, f'{roi:.1f}%', ha='center', va=va, fontsize=10)

plt.tight_layout()
plt.savefig('/Users/mac/Desktop/对抗算法/economic_defense_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存：economic_defense_comparison.png")


# ============================================================================
# 第7部分：总结
# ============================================================================

print("\n\n" + "=" * 80)
print("🎯 核心总结")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────┐
│ 防御效果对比                                                │
├────────────────────────────────────────────────────────────┤
│ 场景       │ 成本   │ 收益   │ 利润    │ ROI     │ 结论   │
├────────────┼────────┼────────┼─────────┼─────────┼────────┤
│ 无防御     │ {costs[0]:>6.0f} │ {revenues[0]:>6.0f} │ +{profits[0]:>6.0f} │ +{rois[0]:>6.1f}% │ 🔥疯狂│
│ 技术防御   │ {costs[1]:>6.0f} │ {revenues[1]:>6.0f} │ +{profits[1]:>6.0f} │ +{rois[1]:>6.1f}% │ 继续干│
│ 经济防御   │ {costs[2]:>6.0f} │ {revenues[2]:>6.0f} │ {profits[2]:>7.0f} │ {rois[2]:>7.1f}% │ ✅放弃│
│ 最优组合   │ {costs[3]:>6.0f} │ {revenues[3]:>6.0f} │ {profits[3]:>7.0f} │ {rois[3]:>7.1f}% │ ✅放弃│
└────────────┴────────┴────────┴─────────┴─────────┴────────┘
""")

print("""
💡 关键洞察：

1. 【技术防御 vs 经济防御】
   
   技术防御：
   - 提升检测率：0% → 50% → 80%
   - 但黑产仍有利可图（利润{:.0f}元）
   - 需要持续投入研发
   - 猫抓老鼠，永无止境
   
   经济防御：
   - 提高成本：{:.0f}元 → {:.0f}元（{:.1f}倍）
   - 降低收益：{:.0f}元 → {:.0f}元（降{:.0f}%）
   - 黑产亏损{:.0f}元，自己放弃
   - 一劳永逸！

2. 【最优策略】
   
   经济手段为核心 + 技术防御为辅助 + 法律威慑
   
   → 成本几乎为0（改规则就行）
   → 效果最好（黑产亏损{:.0f}元）
   → 性价比最高！

3. 【防御哲学】
   
   ✗ 不追求：100%检测所有攻击（做不到）
   ✓ 而是追求：让攻击成本 > 收益
   
   当 利润 < 0 时，黑产自然放弃！

4. 【实战建议】
   
   优先级：
   1. 经济手段（第5-6层）← 最重要！性价比最高
   2. 动态重训练（第3层）← 其次，成本可控
   3. 对抗训练（第2层）← 辅助，提升鲁棒性
   4. 其他技术手段 ← 完善体系

5. 【思维转变】
   
   从技术思维 → 经济思维
   从追求完美 → 追求性价比
   从防御攻击 → 改变博弈规则
   
   这才是真正的工程师思维！

""".format(
    profits[1],  # 技术防御利润
    costs[0], costs[2], costs[2]/costs[0],  # 成本变化
    revenues[0], revenues[2], (revenues[0]-revenues[2])/revenues[0]*100,  # 收益变化
    abs(profits[2]),  # 经济防御亏损
    abs(profits[3])   # 最优组合亏损
))

print("=" * 80)
print("✅ 演示完成！")
print("=" * 80)

print("""
📚 今天学到了什么？

1. ✅ 黑产的决策公式：利润 = 收益 - 成本
2. ✅ 防御的目标：让利润 < 0
3. ✅ 经济手段的威力：釜底抽薪，一劳永逸
4. ✅ 最优策略：经济为核心 + 技术为辅助
5. ✅ 工程师思维：性价比 > 完美主义

下一步：
  → 应用到真实场景（推荐系统、风控、社交网络）
  → 设计完整的防御体系
  → 进入第6关：实战应用！

🎉 恭喜！你已经掌握了图对抗防御的核心思想！
""")

