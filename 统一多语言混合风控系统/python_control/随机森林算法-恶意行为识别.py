#!/usr/bin/env python3

"""
攻击类型识别模块 - 基于随机森林算法
专为感知器系统设计的智能攻击分类器

攻击类型分类：
1. NORMAL - 正常流量
2. CRAWLER - 爬虫攻击
3. BRUTE_FORCE - 暴力破解
4. ORDER_FRAUD - 订单欺诈
5. PAYMENT_FRAUD - 支付欺诈
6. DDoS - 分布式拒绝服务攻击
"""

import time
import numpy as np
import pandas as pd
import pickle
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import warnings
from enum import Enum

# 机器学习库
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

# 导入感知器模块
from 感知器 import KPIData, SceneDetector

warnings.filterwarnings('ignore')


class AttackType(Enum):
    """攻击类型枚举"""
    NORMAL = 0
    CRAWLER = 1
    BRUTE_FORCE = 2
    ORDER_FRAUD = 3
    PAYMENT_FRAUD = 4
    DDOS = 5


@dataclass
class AttackFeatures:
    """攻击特征数据结构"""
    # 基础KPI特征
    order_requests: float
    payment_success: float
    product_pv: float
    risk_hits: float
    
    # 比率特征
    payment_success_rate: float  # 支付成功率
    risk_hit_rate: float        # 风控命中率
    pv_order_ratio: float       # PV与订单比率
    
    # 时间特征
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    is_night: bool
    
    # 异常特征
    time_offset: float
    source_entropy: float
    ip_entropy: float
    
    # 统计特征（滑动窗口）
    order_trend: float          # 订单趋势
    payment_trend: float        # 支付趋势
    pv_trend: float            # PV趋势
    risk_trend: float          # 风控趋势
    
    # 标签
    attack_type: int = AttackType.NORMAL.value


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history_buffer = []
    
    def extract_features(self, kpi_data: KPIData) -> AttackFeatures:
        """从KPI数据提取完整特征"""
        dt = datetime.fromtimestamp(kpi_data.timestamp)
        
        # 计算比率特征
        payment_success_rate = kpi_data.payment_success / max(kpi_data.order_requests, 1)
        risk_hit_rate = kpi_data.risk_hits / max(kpi_data.order_requests, 1)
        pv_order_ratio = kpi_data.product_pv / max(kpi_data.order_requests, 1)
        
        # 时间特征
        hour_of_day = dt.hour
        day_of_week = dt.weekday()
        is_weekend = day_of_week >= 5
        is_night = 0 <= hour_of_day < 6
        
        # 更新历史缓冲区
        self.history_buffer.append({
            'order_requests': kpi_data.order_requests,
            'payment_success': kpi_data.payment_success,
            'product_pv': kpi_data.product_pv,
            'risk_hits': kpi_data.risk_hits
        })
        
        if len(self.history_buffer) > self.window_size:
            self.history_buffer.pop(0)
        
        # 计算趋势特征
        order_trend = self._calculate_trend('order_requests')
        payment_trend = self._calculate_trend('payment_success')
        pv_trend = self._calculate_trend('product_pv')
        risk_trend = self._calculate_trend('risk_hits')
        
        return AttackFeatures(
            order_requests=kpi_data.order_requests,
            payment_success=kpi_data.payment_success,
            product_pv=kpi_data.product_pv,
            risk_hits=kpi_data.risk_hits,
            payment_success_rate=payment_success_rate,
            risk_hit_rate=risk_hit_rate,
            pv_order_ratio=pv_order_ratio,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            is_night=is_night,
            time_offset=abs(kpi_data.time_offset),
            source_entropy=kpi_data.source_entropy,
            ip_entropy=kpi_data.ip_entropy,
            order_trend=order_trend,
            payment_trend=payment_trend,
            pv_trend=pv_trend,
            risk_trend=risk_trend
        )
    
    def _calculate_trend(self, metric: str) -> float:
        """计算趋势（简单线性回归斜率）"""
        if len(self.history_buffer) < 3:
            return 0.0
        
        values = [item[metric] for item in self.history_buffer]
        x = np.arange(len(values))
        y = np.array(values)
        
        # 简单线性回归计算斜率
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 0.0


class AttackDataGenerator:
    """攻击数据生成器 - 用于训练数据合成"""
    
    @staticmethod
    def generate_normal_data(count: int = 1000) -> List[AttackFeatures]:
        """生成正常流量数据"""
        data = []
        for _ in range(count):
            # 正常业务特征
            order_requests = np.random.normal(15, 3)
            payment_success = np.random.normal(12, 2)
            product_pv = np.random.normal(500, 50)
            risk_hits = np.random.normal(2, 1)
            
            features = AttackFeatures(
                order_requests=max(0, order_requests),
                payment_success=max(0, payment_success),
                product_pv=max(0, product_pv),
                risk_hits=max(0, risk_hits),
                payment_success_rate=payment_success / max(order_requests, 1),
                risk_hit_rate=risk_hits / max(order_requests, 1),
                pv_order_ratio=product_pv / max(order_requests, 1),
                hour_of_day=np.random.randint(0, 24),
                day_of_week=np.random.randint(0, 7),
                is_weekend=np.random.choice([True, False]),
                is_night=np.random.choice([True, False]),
                time_offset=abs(np.random.normal(0, 50)),
                source_entropy=np.random.uniform(0.7, 0.95),
                ip_entropy=np.random.uniform(0.8, 0.95),
                order_trend=np.random.normal(0, 0.5),
                payment_trend=np.random.normal(0, 0.3),
                pv_trend=np.random.normal(0, 2),
                risk_trend=np.random.normal(0, 0.2),
                attack_type=AttackType.NORMAL.value
            )
            data.append(features)
        return data
    
    @staticmethod
    def generate_crawler_data(count: int = 200) -> List[AttackFeatures]:
        """生成爬虫攻击数据"""
        data = []
        for _ in range(count):
            # 爬虫特征：高PV，低订单，低支付
            order_requests = np.random.normal(5, 2)
            payment_success = np.random.normal(1, 1)
            product_pv = np.random.normal(2000, 300)  # 异常高PV
            risk_hits = np.random.normal(8, 2)
            
            features = AttackFeatures(
                order_requests=max(0, order_requests),
                payment_success=max(0, payment_success),
                product_pv=max(0, product_pv),
                risk_hits=max(0, risk_hits),
                payment_success_rate=payment_success / max(order_requests, 1),
                risk_hit_rate=risk_hits / max(order_requests, 1),
                pv_order_ratio=product_pv / max(order_requests, 1),  # 高比率
                hour_of_day=np.random.randint(0, 24),
                day_of_week=np.random.randint(0, 7),
                is_weekend=np.random.choice([True, False]),
                is_night=np.random.choice([True, False]),
                time_offset=abs(np.random.normal(0, 100)),
                source_entropy=np.random.uniform(0.2, 0.5),  # 低熵值
                ip_entropy=np.random.uniform(0.3, 0.6),      # 低熵值
                order_trend=np.random.normal(-0.5, 0.3),
                payment_trend=np.random.normal(-0.3, 0.2),
                pv_trend=np.random.normal(5, 2),             # 高增长趋势
                risk_trend=np.random.normal(1, 0.5),
                attack_type=AttackType.CRAWLER.value
            )
            data.append(features)
        return data
    
    @staticmethod
    def generate_brute_force_data(count: int = 150) -> List[AttackFeatures]:
        """生成暴力破解数据"""
        data = []
        for _ in range(count):
            # 暴力破解特征：高订单请求，极低支付成功率，高风控命中
            order_requests = np.random.normal(80, 15)
            payment_success = np.random.normal(2, 1)         # 极低支付成功
            product_pv = np.random.normal(300, 50)
            risk_hits = np.random.normal(25, 5)              # 高风控命中
            
            features = AttackFeatures(
                order_requests=max(0, order_requests),
                payment_success=max(0, payment_success),
                product_pv=max(0, product_pv),
                risk_hits=max(0, risk_hits),
                payment_success_rate=payment_success / max(order_requests, 1),  # 极低成功率
                risk_hit_rate=risk_hits / max(order_requests, 1),               # 高命中率
                pv_order_ratio=product_pv / max(order_requests, 1),
                hour_of_day=np.random.randint(0, 24),
                day_of_week=np.random.randint(0, 7),
                is_weekend=np.random.choice([True, False]),
                is_night=np.random.choice([True, False], p=[0.7, 0.3]),  # 多发生在夜间
                time_offset=abs(np.random.normal(200, 100)),
                source_entropy=np.random.uniform(0.1, 0.3),   # 极低熵值
                ip_entropy=np.random.uniform(0.1, 0.4),       # 极低熵值
                order_trend=np.random.normal(3, 1),           # 订单激增
                payment_trend=np.random.normal(-1, 0.5),      # 支付下降
                pv_trend=np.random.normal(0, 1),
                risk_trend=np.random.normal(2, 0.8),          # 风控上升
                attack_type=AttackType.BRUTE_FORCE.value
            )
            data.append(features)
        return data
    
    @staticmethod
    def generate_order_fraud_data(count: int = 120) -> List[AttackFeatures]:
        """生成订单欺诈数据"""
        data = []
        for _ in range(count):
            # 订单欺诈特征：订单与支付比例异常，中等风控命中
            order_requests = np.random.normal(40, 8)
            payment_success = np.random.normal(35, 5)        # 高支付成功但可疑
            product_pv = np.random.normal(600, 100)
            risk_hits = np.random.normal(12, 3)
            
            features = AttackFeatures(
                order_requests=max(0, order_requests),
                payment_success=max(0, payment_success),
                product_pv=max(0, product_pv),
                risk_hits=max(0, risk_hits),
                payment_success_rate=payment_success / max(order_requests, 1),
                risk_hit_rate=risk_hits / max(order_requests, 1),
                pv_order_ratio=product_pv / max(order_requests, 1),
                hour_of_day=np.random.randint(0, 24),
                day_of_week=np.random.randint(0, 7),
                is_weekend=np.random.choice([True, False]),
                is_night=np.random.choice([True, False]),
                time_offset=abs(np.random.normal(150, 80)),
                source_entropy=np.random.uniform(0.4, 0.7),
                ip_entropy=np.random.uniform(0.5, 0.8),
                order_trend=np.random.normal(2, 0.8),
                payment_trend=np.random.normal(1.5, 0.6),
                pv_trend=np.random.normal(1, 0.5),
                risk_trend=np.random.normal(1, 0.4),
                attack_type=AttackType.ORDER_FRAUD.value
            )
            data.append(features)
        return data
    
    @staticmethod
    def generate_payment_fraud_data(count: int = 100) -> List[AttackFeatures]:
        """生成支付欺诈数据"""
        data = []
        for _ in range(count):
            # 支付欺诈特征：支付异常，高风控命中
            order_requests = np.random.normal(25, 5)
            payment_success = np.random.normal(3, 2)         # 支付异常
            product_pv = np.random.normal(400, 60)
            risk_hits = np.random.normal(18, 4)              # 高风控命中
            
            features = AttackFeatures(
                order_requests=max(0, order_requests),
                payment_success=max(0, payment_success),
                product_pv=max(0, product_pv),
                risk_hits=max(0, risk_hits),
                payment_success_rate=payment_success / max(order_requests, 1),
                risk_hit_rate=risk_hits / max(order_requests, 1),
                pv_order_ratio=product_pv / max(order_requests, 1),
                hour_of_day=np.random.randint(0, 24),
                day_of_week=np.random.randint(0, 7),
                is_weekend=np.random.choice([True, False]),
                is_night=np.random.choice([True, False]),
                time_offset=abs(np.random.normal(300, 150)),
                source_entropy=np.random.uniform(0.3, 0.6),
                ip_entropy=np.random.uniform(0.4, 0.7),
                order_trend=np.random.normal(0.5, 0.5),
                payment_trend=np.random.normal(-1, 0.8),      # 支付下降趋势
                pv_trend=np.random.normal(0, 0.8),
                risk_trend=np.random.normal(1.5, 0.6),        # 风控上升
                attack_type=AttackType.PAYMENT_FRAUD.value
            )
            data.append(features)
        return data
    
    @staticmethod
    def generate_ddos_data(count: int = 80) -> List[AttackFeatures]:
        """生成DDoS攻击数据"""
        data = []
        for _ in range(count):
            # DDoS特征：极高请求量，低成功率，极高PV
            order_requests = np.random.normal(200, 40)       # 极高请求
            payment_success = np.random.normal(1, 1)         # 极低支付
            product_pv = np.random.normal(8000, 1000)        # 极高PV
            risk_hits = np.random.normal(50, 10)             # 极高风控
            
            features = AttackFeatures(
                order_requests=max(0, order_requests),
                payment_success=max(0, payment_success),
                product_pv=max(0, product_pv),
                risk_hits=max(0, risk_hits),
                payment_success_rate=payment_success / max(order_requests, 1),
                risk_hit_rate=risk_hits / max(order_requests, 1),
                pv_order_ratio=product_pv / max(order_requests, 1),
                hour_of_day=np.random.randint(0, 24),
                day_of_week=np.random.randint(0, 7),
                is_weekend=np.random.choice([True, False]),
                is_night=np.random.choice([True, False]),
                time_offset=abs(np.random.normal(500, 200)),
                source_entropy=np.random.uniform(0.05, 0.2),  # 极低熵值
                ip_entropy=np.random.uniform(0.1, 0.3),       # 极低熵值
                order_trend=np.random.normal(10, 3),          # 极高增长
                payment_trend=np.random.normal(-2, 1),        # 支付暴跌
                pv_trend=np.random.normal(20, 5),             # PV暴涨
                risk_trend=np.random.normal(5, 2),            # 风控暴涨
                attack_type=AttackType.DDOS.value
            )
            data.append(features)
        return data


class RandomForestAttackClassifier:
    """随机森林攻击类型分类器"""
    
    def __init__(self, model_path: str = "/tmp/attack_classifier.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'order_requests', 'payment_success', 'product_pv', 'risk_hits',
            'payment_success_rate', 'risk_hit_rate', 'pv_order_ratio',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
            'time_offset', 'source_entropy', 'ip_entropy',
            'order_trend', 'payment_trend', 'pv_trend', 'risk_trend'
        ]
        self.attack_labels = {
            0: "正常流量",
            1: "爬虫攻击", 
            2: "暴力破解",
            3: "订单欺诈",
            4: "支付欺诈",
            5: "DDoS攻击"
        }
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _features_to_array(self, features: AttackFeatures) -> np.ndarray:
        """将特征对象转换为数组"""
        return np.array([
            features.order_requests,
            features.payment_success,
            features.product_pv,
            features.risk_hits,
            features.payment_success_rate,
            features.risk_hit_rate,
            features.pv_order_ratio,
            features.hour_of_day,
            features.day_of_week,
            int(features.is_weekend),
            int(features.is_night),
            features.time_offset,
            features.source_entropy,
            features.ip_entropy,
            features.order_trend,
            features.payment_trend,
            features.pv_trend,
            features.risk_trend
        ]).reshape(1, -1)
    
    def generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成训练数据"""
        self.logger.info("生成训练数据...")
        
        # 生成各类型攻击数据
        normal_data = AttackDataGenerator.generate_normal_data(1000)
        crawler_data = AttackDataGenerator.generate_crawler_data(200)
        brute_force_data = AttackDataGenerator.generate_brute_force_data(150)
        order_fraud_data = AttackDataGenerator.generate_order_fraud_data(120)
        payment_fraud_data = AttackDataGenerator.generate_payment_fraud_data(100)
        ddos_data = AttackDataGenerator.generate_ddos_data(80)
        
        all_data = (normal_data + crawler_data + brute_force_data + 
                   order_fraud_data + payment_fraud_data + ddos_data)
        
        # 转换为数组格式
        X = []
        y = []
        
        for features in all_data:
            feature_array = self._features_to_array(features).flatten()
            X.append(feature_array)
            y.append(features.attack_type)
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"生成数据集大小: {X.shape}, 类别分布: {np.bincount(y)}")
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray):
        """训练随机森林模型"""
        self.logger.info("开始训练随机森林模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 创建随机森林分类器
        self.model = RandomForestClassifier(
            n_estimators=200,           # 树的数量
            max_depth=15,               # 树的最大深度
            min_samples_split=5,        # 内部节点再划分所需最小样本数
            min_samples_leaf=2,         # 叶子节点最少样本数
            max_features='sqrt',        # 每次分割考虑的特征数
            bootstrap=True,             # 是否使用bootstrap采样
            random_state=42,
            n_jobs=-1,                  # 使用所有CPU核心
            class_weight='balanced'     # 自动平衡类别权重
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 模型评估
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        self.logger.info(f"训练集准确率: {train_score:.4f}")
        self.logger.info(f"测试集准确率: {test_score:.4f}")
        
        # 详细评估报告
        y_pred = self.model.predict(X_test)
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, 
                                  target_names=list(self.attack_labels.values())))
        
        # 特征重要性分析
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n特征重要性排序:")
        print(feature_importance.head(10))
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        self.logger.info(f"5折交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return test_score
    
    def save_model(self):
        """保存模型和预处理器"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'attack_labels': self.attack_labels
        }
        joblib.dump(model_data, self.model_path)
        self.logger.info(f"模型已保存到: {self.model_path}")
    
    def load_model(self):
        """加载模型和预处理器"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.attack_labels = model_data['attack_labels']
            self.logger.info(f"模型已从{self.model_path}加载")
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
    def predict_attack_type(self, features: AttackFeatures) -> Tuple[int, str, float]:
        """预测攻击类型"""
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        # 特征转换和标准化
        X = self._features_to_array(features)
        X_scaled = self.scaler.transform(X)
        
        # 预测
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = max(probabilities)
        
        attack_name = self.attack_labels[prediction]
        
        return prediction, attack_name, confidence
    
    def get_attack_probabilities(self, features: AttackFeatures) -> Dict[str, float]:
        """获取各攻击类型的概率"""
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        X = self._features_to_array(features)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return {
            self.attack_labels[i]: prob 
            for i, prob in enumerate(probabilities)
        }


class AttackClassifierIntegrator:
    """攻击分类器与感知器集成器"""
    
    def __init__(self, detector: SceneDetector, classifier: RandomForestAttackClassifier):
        self.detector = detector
        self.classifier = classifier
        self.feature_extractor = FeatureExtractor()
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def enhanced_process_kpi_data(self, kpi_data: KPIData) -> Dict[str, Any]:
        """增强的KPI数据处理 - 集成攻击类型识别"""
        # 原始感知器处理
        detector_result = self.detector.process_kpi_data(kpi_data)
        
        # 特征提取
        features = self.feature_extractor.extract_features(kpi_data)
        
        # 攻击类型预测
        try:
            attack_type, attack_name, confidence = self.classifier.predict_attack_type(features)
            attack_probabilities = self.classifier.get_attack_probabilities(features)
            
            # 构建增强结果
            enhanced_result = {
                'detector_result': detector_result,
                'attack_classification': {
                    'attack_type': attack_type,
                    'attack_name': attack_name,
                    'confidence': confidence,
                    'probabilities': attack_probabilities
                },
                'features': features,
                'timestamp': kpi_data.timestamp
            }
            
            # 根据攻击类型调整策略建议
            if attack_type != AttackType.NORMAL.value:
                enhanced_result['enhanced_policy'] = self._get_attack_specific_policy(
                    attack_type, confidence, detector_result.state
                )
                
                self.logger.warning(
                    f"检测到{attack_name}攻击 (置信度: {confidence:.3f}), "
                    f"场景状态: {detector_result.state.value}"
                )
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"攻击分类失败: {e}")
            return {
                'detector_result': detector_result,
                'attack_classification': None,
                'features': features,
                'timestamp': kpi_data.timestamp
            }
    
    def _get_attack_specific_policy(self, attack_type: int, confidence: float, 
                                  current_state) -> str:
        """根据攻击类型获取特定策略"""
        policies = {
            AttackType.CRAWLER.value: "启用爬虫检测模式：限制单IP请求频率，启用验证码挑战，监控User-Agent",
            AttackType.BRUTE_FORCE.value: "启用暴力破解防护：账户锁定机制，限制登录失败次数，强制验证码",
            AttackType.ORDER_FRAUD.value: "启用订单欺诈防护：订单二次验证，可疑行为检测，人工审核",
            AttackType.PAYMENT_FRAUD.value: "启用支付欺诈防护：支付风险评分模型，交易限额，实名验证",
            AttackType.DDOS.value: "启用DDoS防护：流量限制，IP黑名单，CDN防护",
            AttackType.NORMAL.value: "维持标准安全策略：常规监控和基础防护"
        }
        
        policy = policies.get(attack_type, "未知攻击类型，启用通用防护策略")
        
        # 根据置信度调整策略强度
        if confidence > 0.8:
            policy += "；高置信度，立即执行严格防护措施"
        elif confidence > 0.6:
            policy += "；中等置信度，启用增强监控"
        else:
            policy += "；低置信度，保持观察状态"
            
        return policy


def main():
    """主函数 - 演示随机森林攻击分类器"""
    print("🌲 随机森林攻击分类器 v1.0")
    print("="*60)
    
    # 创建基础组件
    detector = SceneDetector()
    classifier = RandomForestAttackClassifier()
    
    # 创建增强型感知器
    enhanced_detector = AttackClassifierIntegrator(detector, classifier)
    
    print("\n📋 正在生成训练数据...")
    
    # 生成训练数据
    print("生成正常流量数据...")
    normal_data = AttackDataGenerator.generate_normal_data(500)
    print("生成爬虫攻击数据...")
    crawler_data = AttackDataGenerator.generate_crawler_data(100)
    print("生成暴力破解数据...")
    brute_force_data = AttackDataGenerator.generate_brute_force_data(80)
    print("生成订单欺诈数据...")
    order_fraud_data = AttackDataGenerator.generate_order_fraud_data(60)
    print("生成支付欺诈数据...")
    payment_fraud_data = AttackDataGenerator.generate_payment_fraud_data(50)
    print("生成DDoS攻击数据...")
    ddos_data = AttackDataGenerator.generate_ddos_data(40)
    
    # 合并数据
    all_data = normal_data + crawler_data + brute_force_data + order_fraud_data + payment_fraud_data + ddos_data
    
    # 训练模型
    print(f"\n🎯 开始训练模型（数据量: {len(all_data)}）...")
    
    # 转换为数组格式
    X = []
    y = []
    
    for features in all_data:
        feature_array = classifier._features_to_array(features).flatten()
        X.append(feature_array)
        y.append(features.attack_type)
    
    X = np.array(X)
    y = np.array(y)
    
    # 训练模型
    accuracy = classifier.train_model(X, y)
    print(f"✅ 模型训练完成！准确率: {accuracy:.3f}")
    
    print("\n🚀 模拟实时攻击检测...")
    
    # 模拟一些攻击场景
    attack_scenarios = [
        ("正常流量", KPIData(time.time(), 15, 12, 500, 2, 0, 0.8, 0.9)),
        ("爬虫攻击", KPIData(time.time(), 10, 8, 3000, 5, 100, 0.3, 0.4)),
        ("暴力破解", KPIData(time.time(), 80, 5, 400, 25, 200, 0.2, 0.2)),
        ("订单欺诈", KPIData(time.time(), 45, 35, 600, 12, 150, 0.5, 0.6)),
        ("DDoS攻击", KPIData(time.time(), 200, 10, 800, 50, 500, 0.1, 0.1))
    ]
    
    for scenario_name, kpi_data in attack_scenarios:
        print(f"\n📊 {scenario_name}场景检测:")
        result = enhanced_detector.enhanced_process_kpi_data(kpi_data)
        
        print(f"  场景分数: {result['detector_result'].scene_score:.3f}")
        print(f"  系统状态: {result['detector_result'].state.value}")
        
        if result['attack_classification']:
            attack_info = result['attack_classification']
            print(f"  攻击类型: {attack_info['attack_name']} (置信度: {attack_info['confidence']:.3f})")
            if 'enhanced_policy' in result:
                print(f"  增强策略: {result['enhanced_policy']}")
        else:
            print("  攻击类型: 无法分类")
        
        print(f"  基础策略: {result['detector_result'].policy_recommendation}")
    
    print("\n✅ 演示完成！")


if __name__ == "__main__":
    main()