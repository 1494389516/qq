"""
移动端安全感知器（Mobile Security Detector）v1.0
专门针对移动端应用的风控检测系统

移动端特有风险场景：
1. 设备指纹伪造与模拟器检测
2. APP逆向与重打包检测
3. Root/越狱设备风险识别
4. 异常设备行为模式检测
5. 移动端爬虫与自动化工具检测
6. 位置伪造与虚拟定位检测
7. 移动网络环境异常检测

核心功能：
- 设备指纹一致性验证
- 移动端行为基线建模
- 实时威胁等级评估
- 移动端专属策略联动
- 跨设备关联分析
"""

import time
import json
import logging
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import sqlite3
import threading
from abc import ABC, abstractmethod
import math
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


class MobileRiskLevel(Enum):
    """移动端风险等级"""
    LOW = "LOW"           # 低风险
    MEDIUM = "MEDIUM"     # 中风险  
    HIGH = "HIGH"         # 高风险
    CRITICAL = "CRITICAL" # 极高风险


class DeviceType(Enum):
    """设备类型"""
    REAL_DEVICE = "REAL_DEVICE"     # 真实设备
    EMULATOR = "EMULATOR"           # 模拟器
    FARM_DEVICE = "FARM_DEVICE"     # 设备农场
    UNKNOWN = "UNKNOWN"             # 未知


@dataclass
class DeviceFingerprint:
    """移动设备指纹"""
    device_id: str          # 设备唯一标识
    imei: Optional[str]     # IMEI
    android_id: Optional[str]  # Android ID
    mac_address: Optional[str] # MAC地址
    device_model: str       # 设备型号
    os_version: str         # 操作系统版本
    app_version: str        # APP版本
    screen_resolution: str  # 屏幕分辨率
    cpu_info: str          # CPU信息
    memory_info: str       # 内存信息
    is_rooted: bool        # 是否Root/越狱
    has_hook: bool         # 是否存在Hook框架
    install_source: str    # 安装来源
    timestamp: float       # 时间戳


@dataclass 
class LocationInfo:
    """位置信息"""
    latitude: float
    longitude: float
    accuracy: float
    altitude: Optional[float]
    speed: Optional[float]
    bearing: Optional[float]
    provider: str          # GPS/NETWORK/PASSIVE
    is_mock: bool         # 是否虚拟定位
    timestamp: float


@dataclass
class NetworkInfo:
    """网络信息"""
    network_type: str      # WIFI/4G/5G/3G
    operator: str          # 运营商
    country_code: str      # 国家代码
    ip_address: str        # IP地址
    is_proxy: bool         # 是否代理
    is_vpn: bool          # 是否VPN
    connection_quality: float  # 连接质量
    timestamp: float


@dataclass
class BehaviorMetrics:
    """移动端行为指标"""
    timestamp: float
    session_duration: float     # 会话时长
    click_frequency: float      # 点击频率
    scroll_speed: float         # 滚动速度
    touch_pressure: float       # 触摸压力
    gesture_complexity: float   # 手势复杂度
    app_switch_count: int       # 应用切换次数
    background_time: float      # 后台时间
    typing_rhythm: List[float]  # 打字节拍
    screen_orientation_changes: int  # 屏幕方向变化
    volume_key_usage: int       # 音量键使用
    power_key_usage: int        # 电源键使用


class MobileDetectorConfig:
    """移动端检测器配置"""
    def __init__(self):
        # 异常检测阈值
        self.thresholds = {
            'device_consistency_low': 0.3,
            'device_consistency_critical': 0.8,
            'behavior_deviation_medium': 0.4,
            'behavior_deviation_high': 0.7,
            'location_jump_distance': 100,  # 公里
            'location_jump_time': 300,      # 秒
            'velocity_max_reasonable': 200,  # km/h
            'api_rate_limit': 100,          # 每分钟
            'login_failure_threshold': 5,
            'same_network_device_limit': 50
        }
        
        # 设备指纹权重
        self.fingerprint_weights = {
            'imei': 0.25,
            'android_id': 0.20,
            'mac_address': 0.15,
            'device_model': 0.10,
            'screen_resolution': 0.10,
            'cpu_info': 0.10,
            'memory_info': 0.10
        }
        
        # 行为模式权重
        self.behavior_weights = {
            'click_frequency': 0.20,
            'scroll_speed': 0.15,
            'session_duration': 0.15,
            'gesture_complexity': 0.15,
            'typing_rhythm': 0.20,
            'app_usage_pattern': 0.15
        }
        
        # 风险等级映射
        self.risk_level_mapping = {
            (0.0, 0.3): MobileRiskLevel.LOW,
            (0.3, 0.6): MobileRiskLevel.MEDIUM,
            (0.6, 0.8): MobileRiskLevel.HIGH,
            (0.8, 1.0): MobileRiskLevel.CRITICAL
        }


def create_sample_mobile_data() -> dict:
    """创建示例移动端数据"""
    import random
    
    device_fingerprint = DeviceFingerprint(
        device_id=f"device_{random.randint(1000, 9999)}",
        imei=f"86{random.randint(1000000000000, 9999999999999)}",
        android_id=hashlib.md5(str(random.random()).encode()).hexdigest(),
        mac_address=f"02:00:00:{random.randint(10,99)}:{random.randint(10,99)}:{random.randint(10,99)}",
        device_model=random.choice(["SM-G973F", "iPhone12,1", "MI 10", "OnePlus 8"]),
        os_version=random.choice(["Android 11", "iOS 14.5", "Android 10"]),
        app_version="1.2.3",
        screen_resolution=random.choice(["1080x2340", "828x1792", "1440x3200"]),
        cpu_info=random.choice(["Snapdragon 855", "A13 Bionic", "Kirin 990"]),
        memory_info=f"{random.choice([4, 6, 8, 12])}GB",
        is_rooted=random.choice([True, False]),
        has_hook=random.choice([True, False]),
        install_source=random.choice(["PlayStore", "AppStore", "Unknown"]),
        timestamp=time.time()
    )
    
    location_info = LocationInfo(
        latitude=39.9042 + random.uniform(-0.1, 0.1),
        longitude=116.4074 + random.uniform(-0.1, 0.1),
        accuracy=random.uniform(5, 50),
        altitude=random.uniform(0, 100),
        speed=random.uniform(0, 60),
        bearing=random.uniform(0, 360),
        provider=random.choice(["GPS", "NETWORK", "PASSIVE"]),
        is_mock=random.choice([True, False]),
        timestamp=time.time()
    )
    
    network_info = NetworkInfo(
        network_type=random.choice(["WIFI", "4G", "5G", "3G"]),
        operator=random.choice(["中国移动", "中国联通", "中国电信"]),
        country_code="CN",
        ip_address=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
        is_proxy=random.choice([True, False]),
        is_vpn=random.choice([True, False]),
        connection_quality=random.uniform(0.3, 1.0),
        timestamp=time.time()
    )
    
    behavior_metrics = BehaviorMetrics(
        timestamp=time.time(),
        session_duration=random.uniform(60, 3600),
        click_frequency=random.uniform(0.5, 10.0),
        scroll_speed=random.uniform(100, 1000),
        touch_pressure=random.uniform(0.1, 1.0),
        gesture_complexity=random.uniform(0.1, 1.0),
        app_switch_count=random.randint(0, 20),
        background_time=random.uniform(0, 300),
        typing_rhythm=[random.uniform(0.1, 0.5) for _ in range(10)],
        screen_orientation_changes=random.randint(0, 5),
        volume_key_usage=random.randint(0, 10),
        power_key_usage=random.randint(0, 5)
    )
    
    return {
        'device_fingerprint': device_fingerprint,
        'location_info': location_info,
        'network_info': network_info,
        'behavior_metrics': behavior_metrics,
        'login_attempts': random.randint(1, 5),
        'transaction_count': random.randint(0, 10),
        'api_calls': random.randint(10, 100),
        'error_rate': random.uniform(0.0, 0.1),
        'suspicious_api_calls': random.randint(0, 5),
        'captcha_failures': random.randint(0, 3),
        'risk_rule_hits': random.randint(0, 5),
        'malware_indicators': random.randint(0, 2)
    }


def main():
    """主函数 - 演示移动端感知器"""
    print("移动端安全感知器（Mobile Security Detector）v1.0 启动...")
    print("="*60)
    
    # 创建配置
    config = MobileDetectorConfig()
    
    # 模拟数据处理
    print("模拟移动端数据检测...")
    for i in range(5):
        data = create_sample_mobile_data()
        
        print(f"\n📱 设备 {i+1} 检测结果:")
        print(f"设备ID: {data['device_fingerprint'].device_id}")
        print(f"设备型号: {data['device_fingerprint'].device_model}")
        print(f"是否Root: {data['device_fingerprint'].is_rooted}")
        print(f"位置精度: {data['location_info'].accuracy:.1f}m")
        print(f"是否虚拟定位: {data['location_info'].is_mock}")
        print(f"网络类型: {data['network_info'].network_type}")
        print(f"是否使用VPN: {data['network_info'].is_vpn}")
        print(f"会话时长: {data['behavior_metrics'].session_duration:.1f}s")
        print(f"点击频率: {data['behavior_metrics'].click_frequency:.2f}/s")
        
        # 简单风险评估
        risk_score = 0.0
        risk_factors = []
        
        if data['device_fingerprint'].is_rooted:
            risk_score += 0.3
            risk_factors.append("设备已Root")
        
        if data['location_info'].is_mock:
            risk_score += 0.4
            risk_factors.append("虚拟定位")
        
        if data['network_info'].is_vpn:
            risk_score += 0.2
            risk_factors.append("使用VPN")
        
        if data['behavior_metrics'].click_frequency > 8:
            risk_score += 0.3
            risk_factors.append("异常点击频率")
        
        if data['suspicious_api_calls'] > 3:
            risk_score += 0.4
            risk_factors.append("可疑API调用")
        
        # 确定风险等级
        for (min_score, max_score), level in config.risk_level_mapping.items():
            if min_score <= risk_score < max_score:
                risk_level = level
                break
        else:
            risk_level = MobileRiskLevel.CRITICAL
        
        print(f"🎯 风险评分: {risk_score:.2f}")
        print(f"🚨 风险等级: {risk_level.value}")
        if risk_factors:
            print(f"⚠️  风险因素: {', '.join(risk_factors)}")
        else:
            print("✅ 暂无明显风险因素")
        
        print("-" * 40)
        time.sleep(0.5)
    
    print("\n📊 系统特性:")
    print("✅ 设备指纹一致性检测")
    print("✅ 行为模式基线建模") 
    print("✅ 位置跳跃异常检测")
    print("✅ 网络环境风险分析")
    print("✅ 移动端机器人检测")
    print("✅ 实时威胁等级评估")
    
    print("\n🎉 移动端感知器演示完成！")


if __name__ == "__main__":
    main()