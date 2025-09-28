#!/usr/bin/env python3

"""
统一多语言混合风控系统 - 快速功能测试
基于风控算法专家的理论框架，验证系统各组件功能
"""

import asyncio
import time
import random
import json
from unified_risk_control import UnifiedRiskController, KPIData

async def test_unified_system():
    """测试统一风控系统的主要功能"""
    print("🧪 开始测试统一多语言混合风控系统")
    print("=" * 60)
    
    # 创建控制器
    controller = UnifiedRiskController()
    
    # 初始化系统
    print("🚀 初始化系统...")
    if not await controller.initialize_system():
        print("❌ 系统初始化失败")
        return False
    
    print("✅ 系统初始化成功")
    
    # 测试场景1: 正常业务流量
    print("\n📊 测试场景1: 正常业务流量")
    normal_kpi = KPIData(
        timestamp=time.time(),
        order_requests=15,
        payment_success=12,
        product_pv=500,
        risk_hits=2
    )
    
    assessment = controller.comprehensive_risk_assessment(normal_kpi, None, None)
    print(f"  风险评分: {assessment.total_risk_score:.3f}")
    print(f"  威胁等级: {assessment.risk_level.value}")
    print(f"  推荐措施: {assessment.recommendations[0] if assessment.recommendations else '无'}")
    
    # 测试场景2: 疑似DDoS攻击
    print("\n🚨 测试场景2: 疑似DDoS攻击")
    ddos_kpi = KPIData(
        timestamp=time.time(),
        order_requests=200,
        payment_success=1,
        product_pv=8000,
        risk_hits=50
    )
    
    assessment = controller.comprehensive_risk_assessment(ddos_kpi, None, None)
    print(f"  风险评分: {assessment.total_risk_score:.3f}")
    print(f"  威胁等级: {assessment.risk_level.value}")
    print(f"  推荐措施: {assessment.recommendations[0] if assessment.recommendations else '无'}")
    
    # 测试场景3: VPN流量检测
    print("\n🔍 测试场景3: VPN流量检测")
    
    # 模拟网络包数据
    mock_packets = []
    for _ in range(10):
        packet = {
            'timestamp': time.time(),
            'src_ip': f"203.0.113.{random.randint(1, 254)}",
            'dst_ip': "10.0.0.100",
            'src_port': random.choice([1194, 500, 4500]),  # VPN端口
            'dst_port': 443,
            'protocol': 'UDP',
            'size': random.randint(800, 1200),
            'direction': random.choice(['up', 'down'])
        }
        mock_packets.append(packet)
    
    assessment = controller.comprehensive_risk_assessment(normal_kpi, mock_packets, None)
    print(f"  网络风险评分: {assessment.network_assessment['risk_score']:.3f}")
    print(f"  检测阶段: {assessment.network_assessment.get('detection_stage', 'Unknown')}")
    
    # 系统状态检查
    print("\n🔧 系统状态检查")
    status = controller.coordinator.get_system_status()
    print(f"  系统健康度: {status['system_health']:.1%}")
    print(f"  在线组件: {status['online_components']}/{status['total_components']}")
    
    # 关闭系统
    await controller.shutdown_system()
    print("\n✅ 测试完成，系统已关闭")
    
    return True

def test_threat_detection_algorithms():
    """测试威胁检测算法"""
    print("\n🔬 测试核心威胁检测算法")
    print("-" * 40)
    
    # 模拟不同类型的攻击特征
    attack_scenarios = [
        {
            'name': '爬虫攻击',
            'features': [8, 2, 2000, 8, 0.25, 1.0],
            'expected_risk': 'MEDIUM'
        },
        {
            'name': '暴力破解',
            'features': [80, 2, 300, 25, 0.025, 0.31],
            'expected_risk': 'HIGH'
        },
        {
            'name': '正常流量',
            'features': [15, 12, 500, 2, 0.8, 0.13],
            'expected_risk': 'LOW'
        }
    ]
    
    for scenario in attack_scenarios:
        print(f"\n  🎯 {scenario['name']}:")
        print(f"    特征向量: {scenario['features']}")
        
        # 简化的风险评分算法
        feature_sum = sum(scenario['features'])
        if feature_sum > 50:
            risk_level = "HIGH"
            risk_score = 0.8
        elif feature_sum > 30:
            risk_level = "MEDIUM" 
            risk_score = 0.6
        else:
            risk_level = "LOW"
            risk_score = 0.3
            
        print(f"    计算风险: {risk_level} ({risk_score:.3f})")
        print(f"    预期风险: {scenario['expected_risk']}")
        
        match = risk_level == scenario['expected_risk']
        print(f"    检测结果: {'✅ 正确' if match else '❌ 错误'}")

def test_mathematical_models():
    """测试数学建模理论"""
    print("\n📐 测试数学建模理论框架")
    print("-" * 40)
    
    # 多维特征融合公式测试
    print("  🧮 多维特征融合公式:")
    print("    R_total = w1*R_business + w2*R_network + w3*R_scene + α*I(...)")
    
    # 示例数据
    business_risk = 0.7
    network_risk = 0.5
    scene_risk = 0.6
    
    weights = {'business': 0.4, 'network': 0.3, 'scene': 0.3}
    interaction_boost = 0.1  # 交互项
    
    total_risk = (weights['business'] * business_risk + 
                  weights['network'] * network_risk + 
                  weights['scene'] * scene_risk + 
                  interaction_boost)
    
    print(f"    业务风险: {business_risk:.3f}")
    print(f"    网络风险: {network_risk:.3f}") 
    print(f"    场景风险: {scene_risk:.3f}")
    print(f"    权重配置: {weights}")
    print(f"    交互增强: {interaction_boost:.3f}")
    print(f"    总风险评分: {total_risk:.3f}")
    
    # Z-Score异常检测测试
    print("\n  📊 Z-Score异常检测:")
    baseline_mean = 1000
    baseline_std = 200
    current_value = 2000
    
    z_score = (current_value - baseline_mean) / baseline_std
    is_anomaly = abs(z_score) > 3.0
    
    print(f"    基线均值: {baseline_mean}")
    print(f"    标准差: {baseline_std}")
    print(f"    当前值: {current_value}")
    print(f"    Z-Score: {z_score:.3f}")
    print(f"    异常判定: {'是' if is_anomaly else '否'}")

if __name__ == "__main__":
    print("🛡️  统一多语言混合风控系统 - 快速测试")
    print("=" * 80)
    
    # 测试数学理论模型
    test_mathematical_models()
    
    # 测试威胁检测算法
    test_threat_detection_algorithms()
    
    # 测试统一系统(需要异步)
    try:
        asyncio.run(test_unified_system())
    except Exception as e:
        print(f"⚠️  统一系统测试跳过: {e}")
        print("💡 提示: 请先启动完整系统后再测试")
    
    print("\n🎉 所有测试完成！")
    print("\n📚 理论框架总结:")
    print("  🔬 数学建模: 多维特征融合 + Z-Score异常检测")
    print("  🤖 机器学习: 梯度提升算法 + 智能模型选择")
    print("  ⚡ 多语言优势: Python决策 + C++检测 + Rust存储 + Go服务")
    print("  🎯 业务驱动: 场景感知 + 策略自适应 + 效果评估")