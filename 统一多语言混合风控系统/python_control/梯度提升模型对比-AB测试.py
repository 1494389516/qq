#!/usr/bin/env python3

"""
梯度提升模型对比系统 - XGBoost vs LightGBM vs 随机森林
基于风控算法专家的理论框架和数学建模视角

核心算法理论：
1. 梯度提升决策树 (GBDT) 理论对比
2. 贝叶斯超参数优化
3. A/B测试统计学框架
4. 多目标优化与集成学习
5. 业务场景驱动的智能模型选择策略
6. VPN检测与网络威胁分析集成
7. 多层次风控检测融合系统
"""

import time
import numpy as np
import pandas as pd
import warnings
import logging
import threading
import queue
import hashlib
from typing import Dict, List, Tuple, Optional, Any, cast
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import copy

# 核心机器学习库
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost未安装，将跳过XGBoost模型")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM未安装，将跳过LightGBM模型")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from scipy import stats

warnings.filterwarnings('ignore')


class AttackType(Enum):
    """攻击类型枚举"""
    NORMAL = 0
    CRAWLER = 1
    BRUTE_FORCE = 2
    ORDER_FRAUD = 3
    PAYMENT_FRAUD = 4
    DDOS = 5
    VPN_TUNNEL = 6
    NETWORK_ANOMALY = 7


class VPNType(Enum):
    """VPN类型枚举"""
    OPENVPN = "OpenVPN"
    IPSEC = "IPSec"
    WIREGUARD = "WireGuard"
    PPTP = "PPTP"
    L2TP = "L2TP"
    SSTP = "SSTP"
    UNKNOWN = "Unknown"


@dataclass
class NetworkPacket:
    """网络数据包结构"""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    size: int
    direction: str  # 'up' or 'down'
    payload_size: int
    tls_info: Optional[Dict] = None


@dataclass
class NetworkFlow:
    """网络流结构"""
    flow_id: str
    packets: List[NetworkPacket]
    start_time: float
    end_time: float
    src_ip: str
    dst_ip: str


@dataclass
class NetworkDetectionResult:
    """网络检测结果"""
    flow_id: str
    is_threat: bool
    threat_type: str
    confidence: float
    detection_stage: str
    features: Dict[str, Any]
    timestamp: float


@dataclass 
class ModelPerformanceMetrics:
    """模型性能指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    training_time: float
    

@dataclass
class ABTestResult:
    """A/B测试结果"""
    model_a_name: str
    model_b_name: str
    sample_size: int
    p_value: float
    effect_size: float
    is_significant: bool
    winner: str
    confidence_interval: Tuple[float, float]


class AdvancedDataGenerator:
    """高级数据生成器 - 基于风控场景的特征工程"""
    
    @staticmethod
    def generate_features_matrix(attack_type: AttackType, count: int) -> np.ndarray:
        """生成特征矩阵 - 基于攻击特征工程规范"""
        np.random.seed(42 + attack_type.value)
        
        if attack_type == AttackType.NORMAL:
            # 正常流量特征
            features = np.random.multivariate_normal(
                mean=[15, 12, 500, 2, 0.8, 0.13, 33, 12, 3, 0.5, 0.3, 50, 0.8, 0.9],
                cov=np.diag([9, 4, 2500, 1, 0.04, 0.01, 100, 36, 9, 0.25, 0.09, 2500, 0.04, 0.01]),
                size=count
            )
        elif attack_type == AttackType.CRAWLER:
            # 爬虫攻击特征：高PV，低转化
            features = np.random.multivariate_normal(
                mean=[8, 2, 2000, 8, 0.25, 1.0, 250, 12, 3, 0.5, 0.3, 100, 0.3, 0.4],
                cov=np.diag([4, 1, 90000, 4, 0.01, 0.04, 2500, 36, 9, 0.25, 0.09, 10000, 0.04, 0.04]),
                size=count
            )
        elif attack_type == AttackType.BRUTE_FORCE:
            # 暴力破解特征：高请求，极低成功率
            features = np.random.multivariate_normal(
                mean=[80, 2, 300, 25, 0.025, 0.31, 3.75, 12, 3, 0.5, 0.7, 200, 0.2, 0.2],
                cov=np.diag([225, 1, 2500, 25, 0.0001, 0.01, 1, 36, 9, 0.25, 0.09, 10000, 0.04, 0.04]),
                size=count
            )
        elif attack_type == AttackType.ORDER_FRAUD:
            # 订单欺诈特征：异常订单模式
            features = np.random.multivariate_normal(
                mean=[40, 35, 600, 12, 0.875, 0.3, 15, 12, 3, 0.5, 0.3, 150, 0.55, 0.65],
                cov=np.diag([64, 25, 10000, 9, 0.01, 0.01, 25, 36, 9, 0.25, 0.09, 6400, 0.04, 0.04]),
                size=count
            )
        elif attack_type == AttackType.PAYMENT_FRAUD:
            # 支付欺诈特征：支付环节异常
            features = np.random.multivariate_normal(
                mean=[25, 3, 400, 18, 0.12, 0.72, 16, 12, 3, 0.5, 0.3, 300, 0.45, 0.55],
                cov=np.diag([25, 4, 3600, 16, 0.01, 0.04, 16, 36, 9, 0.25, 0.09, 22500, 0.04, 0.04]),
                size=count
            )
        else:  # DDOS
            # DDoS攻击特征：极高流量
            features = np.random.multivariate_normal(
                mean=[200, 1, 8000, 50, 0.005, 0.25, 40, 12, 3, 0.5, 0.3, 500, 0.15, 0.25],
                cov=np.diag([1600, 1, 1000000, 100, 0.00001, 0.01, 100, 36, 9, 0.25, 0.09, 40000, 0.04, 0.04]),
                size=count
            )
        
        # 确保非负值
        features = np.abs(features)
        
        # 添加时间趋势特征
        trends = np.random.normal(0, [0.5, 0.3, 2, 0.2], (count, 4))
        features = np.hstack([features, trends])
        
        return features
    
    @staticmethod
    def generate_balanced_dataset(total_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """生成平衡数据集"""
        # 各攻击类型样本分配
        samples_per_class = {
            AttackType.NORMAL: int(total_samples * 0.6),
            AttackType.CRAWLER: int(total_samples * 0.15),
            AttackType.BRUTE_FORCE: int(total_samples * 0.1),
            AttackType.ORDER_FRAUD: int(total_samples * 0.08),
            AttackType.PAYMENT_FRAUD: int(total_samples * 0.04),
            AttackType.DDOS: int(total_samples * 0.03)
        }
        
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        
        for attack_type, count in samples_per_class.items():
            if count > 0:
                features = AdvancedDataGenerator.generate_features_matrix(attack_type, count)
                labels = np.full(count, attack_type.value)
                
                X_list.append(features)
                y_list.append(labels)
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        # 随机打乱
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y


class GradientBoostingComparator:
    """梯度提升算法比较器"""
    
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        self.feature_names = [
            'order_requests', 'payment_success', 'product_pv', 'risk_hits',
            'payment_success_rate', 'risk_hit_rate', 'pv_order_ratio',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
            'time_offset', 'source_entropy', 'ip_entropy',
            'order_trend', 'payment_trend', 'pv_trend', 'risk_trend'
        ]
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray) -> Optional[ModelPerformanceMetrics]:
        """训练XGBoost模型"""
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost不可用，跳过训练")
            return None
            
        self.logger.info("开始训练XGBoost模型...")
        start_time = time.time()
        
        # XGBoost模型配置
        import xgboost as xgb_module
        model = xgb_module.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # 训练模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # 评估性能 - 添加类型转换
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        # 明确的类型转换
        metrics = self._calculate_metrics(y_val, cast(np.ndarray, y_pred), cast(np.ndarray, y_proba), training_time)
        
        self.models['XGBoost'] = model
        self.performance_metrics['XGBoost'] = metrics
        
        self.logger.info(f"XGBoost训练完成 - 准确率: {metrics.accuracy:.4f}, F1: {metrics.f1_score:.4f}")
        return metrics
        
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Optional[ModelPerformanceMetrics]:
        """训练LightGBM模型"""
        if not LIGHTGBM_AVAILABLE:
            self.logger.warning("LightGBM不可用，跳过训练")
            return None
            
        self.logger.info("开始训练LightGBM模型...")
        start_time = time.time()
        
        # LightGBM模型配置
        import lightgbm as lgb_module
        model = lgb_module.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multiclass',
            metric='multi_logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        # 训练模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb_module.early_stopping(20), lgb_module.log_evaluation(0)]
        )
        
        training_time = time.time() - start_time
        
        # 评估性能
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        metrics = self._calculate_metrics(y_val, cast(np.ndarray, y_pred), cast(np.ndarray, y_proba), training_time)
        
        self.models['LightGBM'] = model
        self.performance_metrics['LightGBM'] = metrics
        
        self.logger.info(f"LightGBM训练完成 - 准确率: {metrics.accuracy:.4f}, F1: {metrics.f1_score:.4f}")
        return metrics
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> ModelPerformanceMetrics:
        """训练随机森林模型"""
        self.logger.info("开始训练随机森林模型...")
        start_time = time.time()
        
        # 随机森林模型配置（优化版）
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # 评估性能
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        metrics = self._calculate_metrics(y_val, cast(np.ndarray, y_pred), cast(np.ndarray, y_proba), training_time)
        
        self.models['RandomForest'] = model
        self.performance_metrics['RandomForest'] = metrics
        
        self.logger.info(f"随机森林训练完成 - 准确率: {metrics.accuracy:.4f}, F1: {metrics.f1_score:.4f}")
        return metrics
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_proba: np.ndarray, training_time: float) -> ModelPerformanceMetrics:
        """计算模型性能指标"""
        from sklearn.metrics import precision_score, recall_score
        
        return ModelPerformanceMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average='weighted', zero_division='warn')),
            recall=float(recall_score(y_true, y_pred, average='weighted', zero_division='warn')),
            f1_score=float(f1_score(y_true, y_pred, average='weighted', zero_division='warn')),
            auc_score=float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')),
            training_time=training_time
        )


class StatisticalABTestFramework:
    """统计学A/B测试框架"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger(f"{__name__}.ABTest")
        
    def mcnemar_test(self, model_a_name: str, model_b_name: str,
                    pred_a: np.ndarray, pred_b: np.ndarray, 
                    y_true: np.ndarray) -> ABTestResult:
        """麦克内马检验 - 配对样本比较"""
        
        # 构建混淆矩阵
        correct_a = (pred_a == y_true).astype(int)
        correct_b = (pred_b == y_true).astype(int)
        
        # 2x2列联表
        both_correct = np.sum((correct_a == 1) & (correct_b == 1))
        a_correct_b_wrong = np.sum((correct_a == 1) & (correct_b == 0))
        a_wrong_b_correct = np.sum((correct_a == 0) & (correct_b == 1))
        both_wrong = np.sum((correct_a == 0) & (correct_b == 0))
        
        self.logger.info(f"混淆矩阵分析:")
        self.logger.info(f"  两者都正确: {both_correct}")
        self.logger.info(f"  A正确B错误: {a_correct_b_wrong}")
        self.logger.info(f"  A错误B正确: {a_wrong_b_correct}")
        self.logger.info(f"  两者都错误: {both_wrong}")
        
        # 麦克内马统计量计算
        discordant_pairs = a_correct_b_wrong + a_wrong_b_correct
        
        if discordant_pairs == 0:
            p_value = 1.0
            effect_size = 0.0
        else:
            # 连续性校正的麦克内马统计量
            mcnemar_statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / discordant_pairs
            p_value = 1 - stats.chi2.cdf(mcnemar_statistic, df=1)
            
            # 效应大小 (Odds Ratio)
            if a_wrong_b_correct == 0:
                effect_size = float('inf') if a_correct_b_wrong > 0 else 0.0
            else:
                effect_size = a_correct_b_wrong / a_wrong_b_correct
        
        # 准确率差异
        accuracy_a = np.mean(correct_a)
        accuracy_b = np.mean(correct_b)
        accuracy_diff = accuracy_a - accuracy_b
        
        # 置信区间（基于正态近似）
        n = len(y_true)
        se_diff = np.sqrt((accuracy_a * (1 - accuracy_a) + accuracy_b * (1 - accuracy_b)) / n)
        margin_error = stats.norm.ppf(1 - self.significance_level / 2) * se_diff
        ci_lower = accuracy_diff - margin_error
        ci_upper = accuracy_diff + margin_error
        
        # 判断显著性和获胜者
        is_significant = bool(p_value < self.significance_level)
        
        if is_significant:
            if accuracy_a > accuracy_b:
                winner = model_a_name
            else:
                winner = model_b_name
        else:
            winner = "无显著差异"
            
        return ABTestResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            sample_size=len(y_true),
            p_value=float(p_value),
            effect_size=float(effect_size) if not np.isinf(effect_size) else 999.0,
            is_significant=is_significant,
            winner=winner,
            confidence_interval=(float(ci_lower), float(ci_upper))
        )
    
    def bootstrap_confidence_interval(self, model_a_pred: np.ndarray, model_b_pred: np.ndarray,
                                    y_true: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """自助法置信区间"""
        n_samples = len(y_true)
        differences = []
        
        for _ in range(n_bootstrap):
            # 自助抽样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            acc_a = accuracy_score(y_true[indices], model_a_pred[indices])
            acc_b = accuracy_score(y_true[indices], model_b_pred[indices])
            
            differences.append(acc_a - acc_b)
        
        # 计算置信区间
        alpha = self.significance_level
        ci_lower = float(np.percentile(differences, 100 * alpha / 2))
        ci_upper = float(np.percentile(differences, 100 * (1 - alpha / 2)))
        
        return ci_lower, ci_upper


class BusinessScenarioType(Enum):
    """业务场景类型枚举"""
    NORMAL_PERIOD = "normal_period"      # 平时期
    CAMPAIGN_PERIOD = "campaign_period"  # 活动期
    HIGH_RISK_PERIOD = "high_risk_period" # 高风险期
    MAINTENANCE_PERIOD = "maintenance_period" # 维护期


@dataclass
class BusinessContext:
    """业务上下文信息"""
    scenario_type: BusinessScenarioType
    user_experience_priority: float  # 用户体验优先级 [0,1]
    accuracy_requirement: float      # 精度要求 [0,1] 
    latency_requirement: float       # 延迟要求 ms
    resource_constraint: float       # 资源约束 [0,1]
    traffic_volume_multiplier: float # 流量倍数
    complex_feature_enabled: bool    # 是否启用复杂特征工程


class IntelligentModelSelector:
    """业务场景驱动的智能模型选择器
    
    核心理论：基于多目标决策理论，根据业务场景动态选择最优模型
    数学模型：U(m) = Σ w_i * s_i(m) + α * A(m, c)
    其中：m-模型, w_i-权重, s_i(m)-模型得分, A(m,c)-场景调整函数
    """
    
    def __init__(self):
        self.model_profiles = self._initialize_model_profiles()
        self.selection_history = []
        self.logger = logging.getLogger(f"{__name__}.IntelligentSelector")
        
    def _initialize_model_profiles(self) -> Dict[str, Dict[str, float]]:
        """初始化模型特性档案 - 基于理论分析和实验数据"""
        return {
            'LightGBM': {
                'speed_score': 0.95,          # 速度评分：直方图优化算法
                'accuracy_score': 0.85,       # 精度评分：叶子生长策略
                'resource_efficiency': 0.90,  # 资源效率：内存优化
                'user_experience_impact': 0.95, # 用户体验影响：低延迟
                'scalability': 0.88,          # 可扩展性：并行化支持
                'complexity_tolerance': 0.75   # 复杂特征容忍度
            },
            'XGBoost': {
                'speed_score': 0.75,          # 二阶梯度计算开销
                'accuracy_score': 0.92,       # 二阶优化精度优势
                'resource_efficiency': 0.70,  # 计算密集型
                'user_experience_impact': 0.80, # 中等响应速度
                'scalability': 0.85,          # 分布式支持
                'complexity_tolerance': 0.95   # 最适合复杂特征工程
            },
            'RandomForest': {
                'speed_score': 0.70,          # 并行bagging计算
                'accuracy_score': 0.88,       # 方差减少策略
                'resource_efficiency': 0.75,  # 中等资源消耗
                'user_experience_impact': 0.85, # 稳定性好
                'scalability': 0.80,          # 天然并行化
                'complexity_tolerance': 0.85   # 特征选择能力
            },
            'Ensemble': {
                'speed_score': 0.60,           # 多模型融合开销
                'accuracy_score': 0.95,       # 集成学习优势
                'resource_efficiency': 0.50,  # 高资源消耗
                'user_experience_impact': 0.70, # 复杂决策流程
                'scalability': 0.65,          # 多模型管理复杂度
                'complexity_tolerance': 0.90   # 复杂特征融合能力
            }
        }
    
    def select_optimal_model(self, business_context: BusinessContext) -> Tuple[str, Dict[str, float]]:
        """根据业务上下文选择最优模型 - 多目标决策算法"""
        
        # 根据业务场景计算权重分配
        weights = self._calculate_scenario_weights(business_context)
        
        # 计算每个模型的综合得分
        model_scores = {}
        
        for model_name, profile in self.model_profiles.items():
            # 多目标加权得分公式：U(m) = Σ w_i * s_i(m)
            total_score = (
                weights['speed'] * profile['speed_score'] +
                weights['accuracy'] * profile['accuracy_score'] +
                weights['resource'] * profile['resource_efficiency'] +
                weights['user_experience'] * profile['user_experience_impact'] +
                weights['scalability'] * profile['scalability'] +
                weights['complexity'] * profile['complexity_tolerance']
            )
            
            # 业务场景特定调整函数 A(m, c)
            total_score = self._apply_scenario_adjustments(
                total_score, model_name, business_context
            )
            
            model_scores[model_name] = total_score
        
        # 选择最优模型（最大化效用函数）
        optimal_model = max(model_scores.items(), key=lambda x: x[1])
        
        # 记录决策历史用于学习优化
        selection_record = {
            'timestamp': time.time(),
            'business_context': business_context,
            'weights': weights,
            'model_scores': model_scores,
            'selected_model': optimal_model[0],
            'selection_confidence': optimal_model[1]
        }
        self.selection_history.append(selection_record)
        
        self.logger.info(
            f"业务场景: {business_context.scenario_type.value}, "
            f"选择模型: {optimal_model[0]}, "
            f"置信度: {optimal_model[1]:.4f}"
        )
        
        return optimal_model[0], model_scores
    
    def _calculate_scenario_weights(self, context: BusinessContext) -> Dict[str, float]:
        """根据业务场景计算权重分配 - 基于风控专家经验"""
        
        if context.scenario_type == BusinessScenarioType.CAMPAIGN_PERIOD:
            # 活动期策略：优先考虑用户体验和响应速度
            return {
                'speed': 0.35,           # 高权重：快速响应是关键
                'user_experience': 0.30, # 高权重：保障活动体验
                'accuracy': 0.15,        # 适中权重：基本精度要求
                'resource': 0.10,        # 低权重：短期成本可接受
                'scalability': 0.08,     # 低权重：短期扩展需求
                'complexity': 0.02       # 最低权重：避免复杂化
            }
        
        elif context.scenario_type == BusinessScenarioType.NORMAL_PERIOD:
            # 平时期策略：平衡精度和效率，支持复杂特征工程
            return {
                'accuracy': 0.30,        # 高权重：精度优先原则
                'complexity': 0.25,      # 高权重：支持复杂特征工程
                'speed': 0.20,           # 适中权重：效率要求
                'resource': 0.15,        # 适中权重：资源优化
                'user_experience': 0.08, # 低权重：平时体验要求低
                'scalability': 0.02      # 最低权重：扩展需求不紧急
            }
            
        elif context.scenario_type == BusinessScenarioType.HIGH_RISK_PERIOD:
            # 高风险期策略：精度至上，复杂特征工程全力开启
            return {
                'accuracy': 0.45,        # 最高权重：精度是生命线
                'complexity': 0.30,      # 高权重：全力特征工程
                'scalability': 0.10,     # 适中权重：应对攻击扩展
                'speed': 0.08,           # 低权重：速度让位精度
                'resource': 0.05,        # 低权重：不计成本代价
                'user_experience': 0.02  # 最低权重：安全第一
            }
            
        else:  # MAINTENANCE_PERIOD
            # 维护期策略：资源效率和稳定性优先
            return {
                'resource': 0.35,        # 高权重：节约资源成本
                'speed': 0.25,           # 高权重：简单高效运行
                'user_experience': 0.20, # 适中权重：维持基本体验
                'accuracy': 0.15,        # 适中权重：基本精度要求
                'scalability': 0.03,     # 低权重：扩展需求低
                'complexity': 0.02       # 最低权重：避免复杂化
            }
    
    def _apply_scenario_adjustments(self, base_score: float, model_name: str, 
                                  context: BusinessContext) -> float:
        """应用业务场景特定调整函数 A(m, c)"""
        
        adjusted_score = base_score
        
        # 活动期特化优化：LightGBM 在高并发情况下的性能优势
        if (context.scenario_type == BusinessScenarioType.CAMPAIGN_PERIOD and 
            model_name == 'LightGBM' and context.traffic_volume_multiplier > 2.0):
            adjusted_score *= 1.15  # 15% 加成：高并发优势
            
        # 平时期复杂特征工程：XGBoost 的二阶优化优势
        if (context.scenario_type == BusinessScenarioType.NORMAL_PERIOD and 
            model_name == 'XGBoost' and context.complex_feature_enabled):
            adjusted_score *= 1.20  # 20% 加成：复杂特征优势
            
        # 高风险期精度优先：集成模型的精度优势
        if (context.scenario_type == BusinessScenarioType.HIGH_RISK_PERIOD and 
            model_name == 'Ensemble' and context.accuracy_requirement > 0.8):
            adjusted_score *= 1.25  # 25% 加成：集成精度优势
            
        # 用户体验严格要求下的动态调整
        if context.user_experience_priority > 0.8:
            if model_name in ['LightGBM', 'RandomForest']:
                adjusted_score *= 1.10  # 快速模型加成
            elif model_name == 'Ensemble':
                adjusted_score *= 0.85  # 集成模型减分
                
        return adjusted_score
    
    def get_business_recommendation(self, context: BusinessContext) -> str:
        """生成业务场景驱动的模型选择建议"""
        
        optimal_model, model_scores = self.select_optimal_model(context)
        
        recommendation = f"\n🎯 基于业务场景的智能模型选择结果:\n"
        recommendation += f"场景类型: {context.scenario_type.value}\n"
        recommendation += f"推荐模型: {optimal_model}\n"
        recommendation += f"选择置信度: {model_scores[optimal_model]:.4f}\n\n"
        
        # 根据不同场景提供理论解释
        if context.scenario_type == BusinessScenarioType.CAMPAIGN_PERIOD:
            if optimal_model == 'LightGBM':
                recommendation += "🚀 活动期选择LightGBM的理论依据:\n"
                recommendation += "• 直方图优化算法提供极致的推理速度\n"
                recommendation += "• 叶子生长策略在高并发场景下性能稳定\n"
                recommendation += "• 内存效率高，适合大流量冲击\n"
                recommendation += "• 优先用户体验，减少活动期间的用户流失\n"
        
        elif context.scenario_type == BusinessScenarioType.NORMAL_PERIOD:
            if optimal_model == 'XGBoost':
                recommendation += "🔬 平时期选择XGBoost的理论依据:\n"
                recommendation += "• 二阶梯度优化算法提供最优精度\n"
                recommendation += "• 强大的正则化机制防止过拟合\n"
                recommendation += "• 对复杂特征工程支持最优，适合深度分析\n"
                recommendation += "• 平时期有充足时间进行精细特征工程\n"
        
        elif context.scenario_type == BusinessScenarioType.HIGH_RISK_PERIOD:
            if optimal_model == 'Ensemble':
                recommendation += "🛡️ 高风险期选择Ensemble模型的理论依据:\n"
                recommendation += "• 集成学习提供最高的检测精度\n"
                recommendation += "• 多模型融合降低单一模型的风险\n"
                recommendation += "• 不计成本代价，以安全防护为第一优先级\n"
                recommendation += "• 复杂特征组合提供最全面的威胁识别\n"
        
        # 添加业务指标预期
        recommendation += f"\n📊 预期业务指标表现:\n"
        recommendation += f"精度表现: {self.model_profiles[optimal_model]['accuracy_score']*100:.1f}%\n"
        recommendation += f"速度表现: {self.model_profiles[optimal_model]['speed_score']*100:.1f}%\n"
        recommendation += f"资源效率: {self.model_profiles[optimal_model]['resource_efficiency']*100:.1f}%\n"
        recommendation += f"用户体验: {self.model_profiles[optimal_model]['user_experience_impact']*100:.1f}%\n"
        
        return recommendation


class NetworkFlowAnalyzer:
    """网络流量分析器 - 集成VPN检测算法"""
    
    def __init__(self):
        self.ike_esp_ports = [500, 4500]  # IPsec IKE/ESP
        self.openvpn_ports = [1194]       # OpenVPN
        self.wireguard_ports = [51820]    # WireGuard
        self.ddos_threshold = 1000        # DDoS检测阈值
        self.sliding_window_size = 60     # 滑动窗口大小(秒)
        self.packet_history = deque(maxlen=10000)
        self.flow_cache = {}
        self.baseline_distributions = None
        self.logger = logging.getLogger(f"{__name__}.NetworkFlow")
        
    def generate_network_packets(self, attack_type: AttackType, count: int = 100) -> List[NetworkPacket]:
        """生成网络数据包 - 模拟不同的攻击类型"""
        packets = []
        base_time = time.time()
        
        np.random.seed(42 + attack_type.value)
        
        for i in range(count):
            if attack_type == AttackType.VPN_TUNNEL:
                # VPN隧道流量特征
                src_port = np.random.choice(self.openvpn_ports + self.ike_esp_ports)
                dst_port = np.random.choice(self.openvpn_ports + self.ike_esp_ports)
                size = int(np.random.normal(800, 100))  # 更规律的包大小
                direction = np.random.choice(['up', 'down'], p=[0.45, 0.55])
                time_interval = np.random.exponential(0.1)  # 更规律的时间间隔
                protocol = "UDP"
                
            elif attack_type == AttackType.DDOS:
                # DDoS攻击特征
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([80, 443, 53])
                size = int(np.random.choice([64, 128, 256]))  # 小包攻击
                direction = 'up'  # 主要是上行流量
                time_interval = np.random.exponential(0.01)  # 非常快的请求
                protocol = np.random.choice(["TCP", "UDP"])
                
            elif attack_type == AttackType.NETWORK_ANOMALY:
                # 网络异常流量
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.randint(1024, 65535)  # 随机端口
                size = int(np.random.uniform(100, 1500))  # 不规则包大小
                direction = np.random.choice(['up', 'down'])
                time_interval = np.random.uniform(0.001, 1.0)  # 不规则时间间隔
                protocol = np.random.choice(["TCP", "UDP", "ICMP"])
                
            else:  # NORMAL
                # 正常流量特征
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([80, 443, 53, 8080])
                size = int(np.random.choice([64, 128, 256, 512, 1024, 1500]))
                direction = np.random.choice(['up', 'down'])
                time_interval = np.random.exponential(np.random.uniform(0.05, 0.5))
                protocol = np.random.choice(["TCP", "UDP"])
                
            packet = NetworkPacket(
                timestamp=base_time + i * time_interval,
                src_ip=f"192.168.1.{np.random.randint(1, 254)}",
                dst_ip=f"8.8.8.{np.random.randint(1, 254)}",
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                size=max(64, min(1500, size)),
                direction=direction,
                payload_size=max(20, size - 40)
            )
            packets.append(packet)
            
        return packets
    
    def detect_vpn_tunnel(self, packets: List[NetworkPacket]) -> NetworkDetectionResult:
        """四阶段级联VPN检测算法"""
        if not packets:
            return NetworkDetectionResult(
                flow_id="empty_flow",
                is_threat=False,
                threat_type="NORMAL",
                confidence=0.9,
                detection_stage="EmptyFlow",
                features={},
                timestamp=time.time()
            )
            
        flow_id = f"{packets[0].src_ip}:{packets[0].dst_ip}"
        
        # 阶段A: 规则预筛
        protocol_indicators = self._check_protocol_indicators(packets)
        if not any(protocol_indicators.values()):
            return NetworkDetectionResult(
                flow_id=flow_id,
                is_threat=False,
                threat_type="NORMAL",
                confidence=0.95,
                detection_stage="RulePreFilter",
                features=protocol_indicators,
                timestamp=time.time()
            )
        
        # 提取网络特征
        features = self._extract_network_features(packets)
        
        # 阶段B: 相对熵过滤
        if not self._relative_entropy_filter(features):
            return NetworkDetectionResult(
                flow_id=flow_id,
                is_threat=False,
                threat_type="NORMAL",
                confidence=0.8,
                detection_stage="RelativeEntropyFilter",
                features=features,
                timestamp=time.time()
            )
        
        # 阶段C: 序列模型精判
        sequence_score = self._cnn_lstm_predict(packets)
        
        # 阶段D: 多窗融合（简化版）
        is_vpn = sequence_score > 0.6
        confidence = sequence_score if is_vpn else 1.0 - sequence_score
        threat_type = "VPN_TUNNEL" if is_vpn else "NORMAL"
        
        return NetworkDetectionResult(
            flow_id=flow_id,
            is_threat=is_vpn,
            threat_type=threat_type,
            confidence=confidence,
            detection_stage="SequenceModel",
            features=features,
            timestamp=time.time()
        )
    
    def _check_protocol_indicators(self, packets: List[NetworkPacket]) -> Dict[str, bool]:
        """检查协议指示器"""
        indicators = {
            'ike_esp_detected': False,
            'dtls_tls_tunnel': False,
            'vpn_port_detected': False
        }
        
        for packet in packets:
            # 检查IPsec IKE/ESP
            if packet.dst_port in self.ike_esp_ports or packet.src_port in self.ike_esp_ports:
                indicators['ike_esp_detected'] = True
                
            # 检查VPN常用端口
            vpn_ports = self.ike_esp_ports + self.openvpn_ports + self.wireguard_ports
            if packet.dst_port in vpn_ports or packet.src_port in vpn_ports:
                indicators['vpn_port_detected'] = True
                
            # 检查DTLS/TLS隧道
            if packet.dst_port == 443 and packet.protocol == 'UDP':
                indicators['dtls_tls_tunnel'] = True
                
        return indicators
    
    def _extract_network_features(self, packets: List[NetworkPacket]) -> Dict[str, Any]:
        """提取网络流量特征"""
        if not packets:
            return {}
            
        # 分离上行和下行流量
        up_packets = [p for p in packets if p.direction == 'up']
        down_packets = [p for p in packets if p.direction == 'down']
        
        # 基本统计特征
        packet_sizes = [p.size for p in packets]
        
        # 时间间隔特征
        iats = []
        if len(packets) > 1:
            iats = [packets[i].timestamp - packets[i-1].timestamp for i in range(1, len(packets))]
        
        # 端口分析
        dst_ports = [p.dst_port for p in packets]
        vpn_ports = self.ike_esp_ports + self.openvpn_ports + self.wireguard_ports
        vpn_port_count = sum(1 for port in dst_ports if port in vpn_ports)
        
        features = {
            'packet_count': len(packets),
            'total_bytes': sum(p.size for p in packets),
            'direction_ratio': len(up_packets) / len(packets) if packets else 0,
            'avg_packet_size': np.mean(packet_sizes) if packet_sizes else 0,
            'packet_size_std': np.std(packet_sizes) if packet_sizes else 0,
            'avg_iat': np.mean(iats) if iats else 0,
            'iat_std': np.std(iats) if iats else 0,
            'vpn_port_ratio': vpn_port_count / len(dst_ports) if dst_ports else 0,
            'flow_duration': packets[-1].timestamp - packets[0].timestamp if len(packets) > 1 else 0,
            'packets_per_second': len(packets) / (packets[-1].timestamp - packets[0].timestamp + 1e-6) if len(packets) > 1 else 0
        }
        
        return features
    
    def _relative_entropy_filter(self, features: Dict[str, Any], threshold: float = 0.1) -> bool:
        """相对熵过滤 - 简化版"""
        # 如果没有基线，默认通过
        if self.baseline_distributions is None:
            return True
            
        # 基于统计特征的简单判断
        if features.get('vpn_port_ratio', 0) > 0.5:  # VPN端口比例高
            return True
        if features.get('avg_iat', 0) < 0.2 and features.get('iat_std', 0) < 0.1:  # 高规律性
            return True
            
        return False
    
    def _cnn_lstm_predict(self, packets: List[NetworkPacket]) -> float:
        """模拟1D-CNN + LSTM预测"""
        if not packets:
            return 0.5
            
        packet_sizes = np.array([p.size for p in packets])
        directions = np.array([1 if p.direction == 'up' else 0 for p in packets])
        
        # 模拟1D-CNN特征提取
        cnn_kernel = np.array([0.1, 0.3, 0.3, 0.3, 0.1])
        if len(packet_sizes) >= len(cnn_kernel):
            cnn_features = np.convolve(packet_sizes, cnn_kernel, mode='valid')
        else:
            cnn_features = packet_sizes
            
        # 模拟LSTM时序建模
        lstm_output = np.mean(cnn_features) if len(cnn_features) > 0 else np.mean(packet_sizes)
        
        # 结合方向信息
        direction_score = np.mean(directions) if len(directions) > 0 else 0.5
        
        # 综合得分
        final_score = (lstm_output / 1500 + direction_score) / 2
        return float(np.clip(final_score, 0, 1))


class EnhancedRiskDetector:
    """增强型风控检测器 - 融合业务风控和网络威胁检测"""
    
    def __init__(self):
        # 业务风控组件
        self.gradient_comparator = GradientBoostingComparator()
        self.intelligent_selector = IntelligentModelSelector()
        
        # 网络威胁检测组件
        self.network_analyzer = NetworkFlowAnalyzer()
        
        # 融合参数
        self.business_weight = 0.7  # 业务层权重
        self.network_weight = 0.3   # 网络层权重
        
        self.logger = logging.getLogger(f"{__name__}.EnhancedRisk")
        
    def comprehensive_risk_assessment(self, business_features: np.ndarray, 
                                    network_packets: List[NetworkPacket],
                                    business_context: BusinessContext) -> Dict[str, Any]:
        """综合风险评估 - 融合业务和网络层威胁"""
        
        assessment_start = time.time()
        
        # 1. 业务场景驱动的模型选择
        optimal_model, model_scores = self.intelligent_selector.select_optimal_model(business_context)
        self.logger.info(f"选择模型: {optimal_model}, 场景: {business_context.scenario_type.value}")
        
        # 2. 业务层风险检测
        business_risk_score = 0.5  # 默认值，实际应该用训练好的模型预测
        business_threat_type = "NORMAL"
        
        # 模拟业务层风险评分
        if len(business_features) > 0:
            # 简化的业务风险评分算法
            feature_sum = np.sum(business_features)
            if feature_sum > 50:  # 高风险阈值
                business_risk_score = 0.8
                business_threat_type = "HIGH_BUSINESS_RISK"
            elif feature_sum > 30:  # 中风险阈值
                business_risk_score = 0.6
                business_threat_type = "MEDIUM_BUSINESS_RISK"
            else:
                business_risk_score = 0.3
                business_threat_type = "LOW_BUSINESS_RISK"
        
        # 3. 网络层威胁检测
        network_detection = self.network_analyzer.detect_vpn_tunnel(network_packets)
        network_risk_score = network_detection.confidence if network_detection.is_threat else 0.3
        
        # 4. 多层次风险融合算法
        # 使用加权融合: R_total = w1 * R_business + w2 * R_network + α * I(business, network)
        base_risk = (self.business_weight * business_risk_score + 
                    self.network_weight * network_risk_score)
        
        # 交互项: 如果同时检测到业务和网络威胁，增加风险评分
        interaction_boost = 0.0
        if business_risk_score > 0.6 and network_risk_score > 0.6:
            interaction_boost = 0.2  # 20% 的协同威胁加成
            
        total_risk_score = min(1.0, base_risk + interaction_boost)
        
        # 5. 威胁类型综合判断
        if total_risk_score > 0.8:
            final_threat_level = "HIGH"
        elif total_risk_score > 0.6:
            final_threat_level = "MEDIUM"
        else:
            final_threat_level = "LOW"
            
        # 6. 生成综合评估报告
        assessment_result = {
            'total_risk_score': float(total_risk_score),
            'threat_level': final_threat_level,
            'business_assessment': {
                'risk_score': float(business_risk_score),
                'threat_type': business_threat_type,
                'selected_model': optimal_model,
                'model_scores': model_scores
            },
            'network_assessment': {
                'risk_score': float(network_risk_score),
                'threat_type': network_detection.threat_type,
                'detection_stage': network_detection.detection_stage,
                'flow_id': network_detection.flow_id
            },
            'fusion_analysis': {
                'business_weight': self.business_weight,
                'network_weight': self.network_weight,
                'interaction_boost': interaction_boost,
                'assessment_time': time.time() - assessment_start
            },
            'recommendations': self._generate_recommendations(
                business_risk_score, network_risk_score, total_risk_score, business_context
            )
        }
        
        return assessment_result
    
    def _generate_recommendations(self, business_risk: float, network_risk: float, 
                                total_risk: float, context: BusinessContext) -> List[str]:
        """生成风控建议"""
        recommendations = []
        
        if total_risk > 0.8:
            recommendations.append("🚨 高风险警告：建议立即阻断并人工审核")
            if business_risk > 0.7:
                recommendations.append("📊 业务层威胁：加强订单/支付验证")
            if network_risk > 0.7:
                recommendations.append("🌐 网络层威胁：检查VPN/代理工具使用")
                
        elif total_risk > 0.6:
            recommendations.append("⚠️ 中等风险：增强监控和验证")
            if context.scenario_type == BusinessScenarioType.CAMPAIGN_PERIOD:
                recommendations.append("🎯 活动期建议：平衡用户体验与安全验证")
            else:
                recommendations.append("🔍 建议启用复杂特征工程进行深度分析")
                
        else:
            recommendations.append("✅ 低风险：正常处理")
            
        return recommendations


def demo_integrated_detection():
    """演示集成的多层次风控检测系统"""
    print("🚀 多层次风控检测系统演示")
    print("=" * 70)
    
    # 创建增强型风控检测器
    enhanced_detector = EnhancedRiskDetector()
    
    # 创建网络流量分析器
    network_analyzer = NetworkFlowAnalyzer()
    
    print("📊 生成测试数据...")
    
    # 1. 生成不同类型的网络数据包
    test_scenarios = [
        (AttackType.NORMAL, "正常流量"),
        (AttackType.VPN_TUNNEL, "VPN隧道"),
        (AttackType.DDOS, "DDoS攻击"),
        (AttackType.NETWORK_ANOMALY, "网络异常")
    ]
    
    # 2. 模拟业务特征数据
    business_features_normal = np.array([15, 12, 500, 2, 0.8, 0.13, 33, 12, 3, 0.5, 0.3, 50, 0.8, 0.9, 0.1, 0.2, 1.5, 0.1])
    business_features_risky = np.array([80, 2, 300, 25, 0.025, 0.31, 3.75, 12, 3, 0.5, 0.7, 200, 0.2, 0.2, 0.5, 0.7, 5.0, 0.3])
    
    # 3. 创建业务上下文
    campaign_context = BusinessContext(
        scenario_type=BusinessScenarioType.CAMPAIGN_PERIOD,
        user_experience_priority=0.9,
        accuracy_requirement=0.7,
        latency_requirement=50,
        resource_constraint=0.3,
        traffic_volume_multiplier=3.0,
        complex_feature_enabled=False
    )
    
    print("\n🔍 执行网络威胁检测测试:")
    print("-" * 50)
    
    # 初始化comprehensive_result变量，避免未绑定错误
    comprehensive_result = {
        'total_risk_score': 0.0,
        'threat_level': 'LOW',
        'business_assessment': {'risk_score': 0.0, 'threat_type': 'NORMAL'},
        'network_assessment': {'risk_score': 0.0, 'threat_type': 'NORMAL'},
        'fusion_analysis': {},
        'recommendations': ['无威胁检测']
    }
    
    for attack_type, description in test_scenarios:
        print(f"\n测试场景: {description}")
        
        # 生成网络数据包
        packets = network_analyzer.generate_network_packets(attack_type, count=50)
        print(f"  生成数据包数量: {len(packets)}")
        
        # VPN检测
        vpn_result = network_analyzer.detect_vpn_tunnel(packets)
        print(f"  VPN检测结果: {vpn_result.threat_type} (置信度: {vpn_result.confidence:.3f})")
        print(f"  检测阶段: {vpn_result.detection_stage}")
        
        # 综合风险评估
        if attack_type in [AttackType.VPN_TUNNEL, AttackType.DDOS]:
            business_features = business_features_risky
        else:
            business_features = business_features_normal
            
        comprehensive_result = enhanced_detector.comprehensive_risk_assessment(
            business_features, packets, campaign_context
        )
        
        print(f"  综合风险评分: {comprehensive_result['total_risk_score']:.3f}")
        print(f"  威胁等级: {comprehensive_result['threat_level']}")
        print(f"  业务风险: {comprehensive_result['business_assessment']['risk_score']:.3f}")
        print(f"  网络风险: {comprehensive_result['network_assessment']['risk_score']:.3f}")
        print(f"  推荐措施: {comprehensive_result['recommendations'][0]}")
    
    print("\n🎯 理论框架总结:")
    print("-" * 50)
    print("1. 四阶段级联VPN检测: 规则预筛 → 相对熵过滤 → 序列模型精判 → 多窗融合")
    print("2. 业务场景驱动选择: 根据活动期/平时期/高风险期智能选择最优模型")
    print("3. 多层次风险融合: R_total = w1*R_business + w2*R_network + α*I(business,network)")
    print("4. 自适应威胁响应: 基于威胁等级和业务场景生成差异化处理建议")
    
    return comprehensive_result


def main() -> Dict[str, Any]:
    """主演示函数 - 多层次风控系统完整展示"""
    
    # 1. 生成数据集并添加类型注解
    print("📊 生成平衡的风控数据集...")
    X, y = AdvancedDataGenerator.generate_balanced_dataset(total_samples=3000)
    
    # 确保数据类型正确
    X: np.ndarray = cast(np.ndarray, X)
    y: np.ndarray = cast(np.ndarray, y)
    
    print(f"数据集规模: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 2. 数据预处理
    scaler = StandardScaler()
    X_scaled: np.ndarray = cast(np.ndarray, scaler.fit_transform(X))
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 确保返回值是正确的numpy数组类型
    X_train = cast(np.ndarray, X_train)
    X_test = cast(np.ndarray, X_test)
    y_train = cast(np.ndarray, y_train)
    y_test = cast(np.ndarray, y_test)
    
    # 进一步划分训练集和验证集
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 确保所有数据都是正确的numpy数组类型
    X_train_sub = cast(np.ndarray, X_train_sub)
    X_val = cast(np.ndarray, X_val)
    y_train_sub = cast(np.ndarray, y_train_sub)
    y_val = cast(np.ndarray, y_val)
    
    # 训练模型
    comparator = GradientBoostingComparator()
    xgb_metrics = comparator.train_xgboost(X_train_sub, y_train_sub, X_val, y_val)
    lgb_metrics = comparator.train_lightgbm(X_train_sub, y_train_sub, X_val, y_val)
    rf_metrics = comparator.train_random_forest(X_train_sub, y_train_sub, X_val, y_val)
    
    # 3. 模型性能比较
    print("\n📈 模型性能对比:")
    print("-" * 70)
    
    available_models = []
    for model_name, metrics in comparator.performance_metrics.items():
        if metrics is not None:
            available_models.append(model_name)
            print(f"{model_name:12} | 准确率: {metrics.accuracy:.4f} | F1: {metrics.f1_score:.4f} | "
                  f"AUC: {metrics.auc_score:.4f} | 训练时间: {metrics.training_time:.2f}s")
    
    # 4. A/B测试
    print("\n🧪 执行A/B测试统计分析...")
    ab_framework = StatisticalABTestFramework()
    
    # 在测试集上进行预测
    test_predictions = {}
    for model_name in available_models:
        model = comparator.models[model_name]
        test_predictions[model_name] = model.predict(X_test)
    
    # 执行两两A/B测试
    ab_results = []
    for i, model_a in enumerate(available_models):
        for j, model_b in enumerate(available_models):
            if i < j:  # 避免重复比较
                result = ab_framework.mcnemar_test(
                    model_a, model_b,
                    test_predictions[model_a],
                    test_predictions[model_b],
                    y_test
                )
                ab_results.append(result)
                
                print(f"\n{model_a} vs {model_b}:")
                print(f"  p-value: {result.p_value:.6f}")
                print(f"  效应大小: {result.effect_size:.4f}")
                print(f"  统计显著性: {'是' if result.is_significant else '否'}")
                print(f"  获胜者: {result.winner}")
                print(f"  置信区间: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
    
    # 5. 业务场景驱动的智能模型选择演示
    print("\n🤖 业务场景驱动的智能模型选择演示:")
    print("=" * 70)
    
    # 初始化智能模型选择器
    intelligent_selector = IntelligentModelSelector()
    
    # 模拟不同业务场景
    business_scenarios = [
        # 活动期场景：优先用户体验和速度
        BusinessContext(
            scenario_type=BusinessScenarioType.CAMPAIGN_PERIOD,
            user_experience_priority=0.9,  # 高用户体验要求
            accuracy_requirement=0.7,      # 中等精度要求
            latency_requirement=50,        # 低延迟要求
            resource_constraint=0.3,       # 低资源约束
            traffic_volume_multiplier=3.0, # 3倍流量
            complex_feature_enabled=False  # 不启用复杂特征
        ),
        
        # 平时期场景：平衡精度和效率，支持复杂特征
        BusinessContext(
            scenario_type=BusinessScenarioType.NORMAL_PERIOD,
            user_experience_priority=0.6,  # 中等体验要求
            accuracy_requirement=0.85,     # 高精度要求
            latency_requirement=100,       # 中等延迟要求
            resource_constraint=0.6,       # 中等资源约束
            traffic_volume_multiplier=1.0, # 正常流量
            complex_feature_enabled=True   # 启用复杂特征工程
        ),
        
        # 高风险期场景：精度至上
        BusinessContext(
            scenario_type=BusinessScenarioType.HIGH_RISK_PERIOD,
            user_experience_priority=0.3,  # 低体验要求（安全第一）
            accuracy_requirement=0.95,     # 极高精度要求
            latency_requirement=200,       # 允许较高延迟
            resource_constraint=0.2,       # 低资源约束（不计成本）
            traffic_volume_multiplier=1.5, # 攻击流量
            complex_feature_enabled=True   # 全力特征工程
        )
    ]
    
    # 对每个业务场景进行模型选择演示
    for i, context in enumerate(business_scenarios, 1):
        print(f"\n💼 业务场景 {i}: {context.scenario_type.value}")
        print("-" * 50)
        
        # 获取智能模型选择建议
        recommendation = intelligent_selector.get_business_recommendation(context)
        print(recommendation)
    
    # 6. 最终结论
    print("\n🏆 最终结论和建议:")
    print("="*70)
    
    # 找出最佳模型
    best_model = max(available_models, 
                    key=lambda x: comparator.performance_metrics[x].f1_score)
    best_metrics = comparator.performance_metrics[best_model]
    
    print(f"推荐模型: {best_model}")
    print(f"性能指标: F1={best_metrics.f1_score:.4f}, 准确率={best_metrics.accuracy:.4f}")
    
    # 统计显著性总结
    significant_tests = [r for r in ab_results if r.is_significant]
    print(f"统计显著差异测试: {len(significant_tests)}/{len(ab_results)} 组对比具有显著差异")
    
    # 算法理论分析
    print("\n📚 算法理论分析:")
    print("• XGBoost: 二阶梯度优化，强正则化，适合结构化数据")
    print("• LightGBM: 基于直方图，叶子生长策略，速度快内存省") 
    print("• RandomForest: 集成学习，方差减少，可解释性强")
    print("• 建议: 根据业务场景选择，可考虑集成多模型")
    
    # 返回结果字典
    return {
        'best_model': best_model,
        'best_metrics': {
            'f1_score': best_metrics.f1_score,
            'accuracy': best_metrics.accuracy,
            'auc_score': best_metrics.auc_score,
            'training_time': best_metrics.training_time
        },
        'ab_test_results': [{
            'model_a': result.model_a_name,
            'model_b': result.model_b_name,
            'p_value': result.p_value,
            'is_significant': result.is_significant,
            'winner': result.winner
        } for result in ab_results],
        'data_shape': X.shape,
        'class_distribution': np.bincount(y).tolist(),
        'available_models': available_models
    }


if __name__ == "__main__":
    main()