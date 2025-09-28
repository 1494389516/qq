

"""
风控模型优化系统 - 随机森林增强 + LightGBM集成 + A/B测试框架
基于风控算法专家的数学建模和理论分析视角

核心优化理论：
1. 随机森林参数优化与特征工程
2. LightGBM梯度提升集成
3. 贝叶斯A/B测试统计框架
4. 多目标优化与模型解释性
"""

import time
import numpy as np
import pandas as pd
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any, cast
from dataclasses import dataclass
from enum import Enum

# 核心机器学习库
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from scipy import stats
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# LightGBM (如果可用)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM未安装，将仅使用随机森林和极端随机树")

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
class ModelPerformanceMetrics:
    """模型性能指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    training_time: float
    prediction_time: float
    feature_importance: Dict[str, float]
    

@dataclass
class ABTestResult:
    """A/B测试结果"""
    model_a_name: str
    model_b_name: str
    sample_size: int
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    winner: str
    business_impact: Dict[str, float]


class AdvancedFeatureEngineering:
    """高级特征工程 - 基于风控场景的特征构造"""
    
    def __init__(self):
        self.feature_names = [
            # 基础KPI特征
            'order_requests', 'payment_success', 'product_pv', 'risk_hits',
            # 比率特征 
            'payment_success_rate', 'risk_hit_rate', 'pv_order_ratio',
            # 时间特征
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
            # 熵值特征
            'time_offset', 'source_entropy', 'ip_entropy',
            # 趋势特征
            'order_trend', 'payment_trend', 'pv_trend', 'risk_trend',
            # 高级组合特征
            'risk_payment_interaction', 'temporal_risk_score', 'behavior_consistency'
        ]
    
    @staticmethod
    def generate_enhanced_features(attack_type: AttackType, count: int) -> np.ndarray:
        """生成增强特征矩阵 - 基于攻击特征工程规范"""
        np.random.seed(42 + attack_type.value)
        
        # 基础特征生成
        if attack_type == AttackType.NORMAL:
            # 正常流量：平衡的业务指标
            base_features = np.random.multivariate_normal(
                mean=[15, 12, 500, 2, 0.8, 0.13, 33, 12, 3, 0.5, 0.3, 50, 0.8, 0.9],
                cov=np.diag([9, 4, 2500, 1, 0.04, 0.01, 100, 36, 9, 0.25, 0.09, 2500, 0.04, 0.01]),
                size=count
            )
        elif attack_type == AttackType.CRAWLER:
            # 爬虫攻击：高PV，低转化，异常熵值
            base_features = np.random.multivariate_normal(
                mean=[8, 2, 2000, 8, 0.25, 1.0, 250, 12, 3, 0.5, 0.3, 100, 0.3, 0.4],
                cov=np.diag([4, 1, 90000, 4, 0.01, 0.04, 2500, 36, 9, 0.25, 0.09, 10000, 0.04, 0.04]),
                size=count
            )
        elif attack_type == AttackType.BRUTE_FORCE:
            # 暴力破解：高频请求，极低成功率
            base_features = np.random.multivariate_normal(
                mean=[80, 2, 300, 25, 0.025, 0.31, 3.75, 12, 3, 0.5, 0.7, 200, 0.2, 0.2],
                cov=np.diag([225, 1, 2500, 25, 0.0001, 0.01, 1, 36, 9, 0.25, 0.09, 10000, 0.04, 0.04]),
                size=count
            )
        elif attack_type == AttackType.ORDER_FRAUD:
            # 订单欺诈：订单支付模式异常
            base_features = np.random.multivariate_normal(
                mean=[40, 35, 600, 12, 0.875, 0.3, 15, 12, 3, 0.5, 0.3, 150, 0.55, 0.65],
                cov=np.diag([64, 25, 10000, 9, 0.01, 0.01, 25, 36, 9, 0.25, 0.09, 6400, 0.04, 0.04]),
                size=count
            )
        elif attack_type == AttackType.PAYMENT_FRAUD:
            # 支付欺诈：支付环节高风险
            base_features = np.random.multivariate_normal(
                mean=[25, 3, 400, 18, 0.12, 0.72, 16, 12, 3, 0.5, 0.3, 300, 0.45, 0.55],
                cov=np.diag([25, 4, 3600, 16, 0.01, 0.04, 16, 36, 9, 0.25, 0.09, 22500, 0.04, 0.04]),
                size=count
            )
        else:  # DDOS
            # DDoS攻击：极高流量冲击
            base_features = np.random.multivariate_normal(
                mean=[200, 1, 8000, 50, 0.005, 0.25, 40, 12, 3, 0.5, 0.3, 500, 0.15, 0.25],
                cov=np.diag([1600, 1, 1000000, 100, 0.00001, 0.01, 100, 36, 9, 0.25, 0.09, 40000, 0.04, 0.04]),
                size=count
            )
        
        # 确保非负值
        base_features = np.abs(base_features)
        
        # 添加趋势特征
        trends = np.random.normal(0, [0.5, 0.3, 2, 0.2], (count, 4))
        
        # 构造高级组合特征
        advanced_features = np.zeros((count, 3))
        
        # 风险-支付交互特征
        advanced_features[:, 0] = base_features[:, 3] * (1 - base_features[:, 4])  # risk_hits * (1 - payment_rate)
        
        # 时间风险分数 
        night_factor = base_features[:, 10] * 1.5 + 1  # 夜间加权
        advanced_features[:, 1] = base_features[:, 3] * night_factor
        
        # 行为一致性分数
        expected_pv = base_features[:, 0] * 30  # 预期PV
        pv_deviation = np.abs(base_features[:, 2] - expected_pv) / (expected_pv + 1)
        advanced_features[:, 2] = 1 / (1 + pv_deviation)  # 一致性分数
        
        # 合并所有特征
        features = np.hstack([base_features, trends, advanced_features])
        
        return features
    
    @staticmethod
    def generate_balanced_dataset(total_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
        """生成平衡的风控数据集"""
        # 各攻击类型样本分配 (符合真实业务分布)
        samples_per_class = {
            AttackType.NORMAL: int(total_samples * 0.65),      # 正常流量占主导
            AttackType.CRAWLER: int(total_samples * 0.15),     # 爬虫攻击较常见
            AttackType.BRUTE_FORCE: int(total_samples * 0.08), # 暴力破解
            AttackType.ORDER_FRAUD: int(total_samples * 0.06), # 订单欺诈
            AttackType.PAYMENT_FRAUD: int(total_samples * 0.04), # 支付欺诈
            AttackType.DDOS: int(total_samples * 0.02)         # DDoS攻击较少
        }
        
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        
        for attack_type, count in samples_per_class.items():
            if count > 0:
                features = AdvancedFeatureEngineering.generate_enhanced_features(attack_type, count)
                labels = np.full(count, attack_type.value)
                
                X_list.append(features)
                y_list.append(labels)
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        # 随机打乱数据
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y


class OptimizedRandomForestModel:
    """优化的随机森林模型"""
    
    def __init__(self, optimization_level: str = 'advanced'):
        self.optimization_level = optimization_level
        self.model = None
        self.feature_names = AdvancedFeatureEngineering().feature_names
        self.training_metrics = None
        
        # 日志配置
        self.logger = logging.getLogger(f"{__name__}.RandomForest")
    
    def train_with_hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                                        X_val: np.ndarray, y_val: np.ndarray) -> ModelPerformanceMetrics:
        """带超参数调优的训练"""
        self.logger.info(f"开始训练优化随机森林模型 (优化级别: {self.optimization_level})...")
        start_time = time.time()
        
        if self.optimization_level == 'basic':
            # 基础配置
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
        elif self.optimization_level == 'advanced':
            # 高级配置
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [12, 15, 18],
                'min_samples_split': [5, 8, 12],
                'min_samples_leaf': [2, 3, 5],
                'max_features': ['sqrt', 'log2', 0.8]
                
            }
        else:  # expert
            # 专家级配置
            param_grid = {
                'n_estimators': [300, 400, 500],
                'max_depth': [15, 18, 22],
                'min_samples_split': [5, 8, 12, 15],
                'min_samples_leaf': [2, 3, 5, 8],
                'max_features': ['sqrt', 'log2', 0.6, 0.8, 1.0],
                'min_impurity_decrease': [0.0, 0.001, 0.01]
            }
        
        # 网格搜索优化
        base_model = RandomForestClassifier(
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            criterion='gini'
        )
        
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        training_time = time.time() - start_time
        
        # 评估性能
        start_pred_time = time.time()
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)
        prediction_time = time.time() - start_pred_time
        
        # 特征重要性
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        metrics = ModelPerformanceMetrics(
            accuracy=float(accuracy_score(y_val, y_pred)),
            precision=float(precision_score(y_val, y_pred, average='weighted', zero_division='warn')),
            recall=float(recall_score(y_val, y_pred, average='weighted', zero_division='warn')),
            f1_score=float(f1_score(y_val, y_pred, average='weighted', zero_division='warn')),
            auc_score=float(roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')),
            training_time=training_time,
            prediction_time=prediction_time,
            feature_importance=feature_importance
        )
        
        self.training_metrics = metrics
        
        self.logger.info(f"随机森林训练完成 - F1: {metrics.f1_score:.4f}, 最佳参数: {grid_search.best_params_}")
        return metrics


class LightGBMModel:
    """LightGBM模型"""
    
    def __init__(self):
        self.model = None
        self.feature_names = AdvancedFeatureEngineering().feature_names
        self.training_metrics = None
        self.logger = logging.getLogger(f"{__name__}.LightGBM")
    
    def train_optimized(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> Optional[ModelPerformanceMetrics]:
        """训练优化的LightGBM模型"""
        if not LIGHTGBM_AVAILABLE:
            self.logger.warning("LightGBM不可用，跳过训练")
            return None
            
        self.logger.info("开始训练LightGBM模型...")
        start_time = time.time()
        
        # LightGBM优化配置 - 确保lgb已导入
        # 导入lgb模块的引用确保可用
        import lightgbm as lgb_module
        
        model = lgb_module.LGBMClassifier(
            objective='multiclass',
            num_class=6,
            metric='multi_logloss',
            boosting_type='gbdt',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=12,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            min_child_weight=0.001,
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
        
        self.model = model
        training_time = time.time() - start_time
        
        # 评估性能
        start_pred_time = time.time()
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        prediction_time = time.time() - start_pred_time
        
        # 特征重要性
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        metrics = ModelPerformanceMetrics(
            accuracy=float(accuracy_score(y_val, y_pred)),
            precision=float(precision_score(y_val, y_pred, average='weighted', zero_division='warn')),
            recall=float(recall_score(y_val, y_pred, average='weighted', zero_division='warn')),
            f1_score=float(f1_score(y_val, y_pred, average='weighted', zero_division='warn')),
            auc_score=float(roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')),
            training_time=training_time,
            prediction_time=prediction_time,
            feature_importance=feature_importance
        )
        
        self.training_metrics = metrics
        
        self.logger.info(f"LightGBM训练完成 - F1: {metrics.f1_score:.4f}")
        return metrics


class BayesianABTestFramework:
    """贝叶斯A/B测试框架"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger(f"{__name__}.BayesianABTest")
    
    def mcnemar_test_with_business_metrics(self, model_a_name: str, model_b_name: str,
                                         pred_a: np.ndarray, pred_b: np.ndarray,
                                         y_true: np.ndarray) -> ABTestResult:
        """麦克内马检验加业务指标分析"""
        
        # 基础统计检验
        correct_a = (pred_a == y_true).astype(int)
        correct_b = (pred_b == y_true).astype(int)
        
        # 混淆矩阵
        both_correct = np.sum((correct_a == 1) & (correct_b == 1))
        a_correct_b_wrong = np.sum((correct_a == 1) & (correct_b == 0))
        a_wrong_b_correct = np.sum((correct_a == 0) & (correct_b == 1))
        both_wrong = np.sum((correct_a == 0) & (correct_b == 0))
        
        # 麦克内马统计量
        discordant_pairs = a_correct_b_wrong + a_wrong_b_correct
        
        if discordant_pairs == 0:
            p_value = 1.0
            effect_size = 0.0
        else:
            mcnemar_statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / discordant_pairs
            p_value = 1 - stats.chi2.cdf(mcnemar_statistic, df=1)
            effect_size = a_correct_b_wrong / a_wrong_b_correct if a_wrong_b_correct > 0 else float('inf')
        
        # 准确率差异和置信区间
        accuracy_a = np.mean(correct_a)
        accuracy_b = np.mean(correct_b)
        accuracy_diff = accuracy_a - accuracy_b
        
        n = len(y_true)
        se_diff = np.sqrt((accuracy_a * (1 - accuracy_a) + accuracy_b * (1 - accuracy_b)) / n)
        margin_error = stats.norm.ppf(1 - self.significance_level / 2) * se_diff
        ci_lower = accuracy_diff - margin_error
        ci_upper = accuracy_diff + margin_error
        
        # 业务影响分析
        business_impact = self._calculate_business_impact(pred_a, pred_b, y_true)
        
        # 判断获胜者
        is_significant_bool = bool(p_value < self.significance_level)
        if is_significant_bool:
            winner = model_a_name if accuracy_a > accuracy_b else model_b_name
        else:
            winner = "无显著差异"
        
        return ABTestResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            sample_size=len(y_true),
            p_value=float(p_value),
            effect_size=float(effect_size) if not np.isinf(effect_size) else 999.0,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            is_significant=is_significant_bool,
            winner=winner,
            business_impact=business_impact
        )
    
    def _calculate_business_impact(self, pred_a: np.ndarray, pred_b: np.ndarray, 
                                  y_true: np.ndarray) -> Dict[str, float]:
        """计算业务影响指标"""
        
        # 假阳性率 (正常流量被误判为攻击)
        normal_mask = (y_true == 0)
        if np.sum(normal_mask) > 0:
            fp_rate_a = np.sum(pred_a[normal_mask] != 0) / np.sum(normal_mask)
            fp_rate_b = np.sum(pred_b[normal_mask] != 0) / np.sum(normal_mask)
            fp_improvement = (fp_rate_a - fp_rate_b) / fp_rate_a * 100 if fp_rate_a > 0 else 0
        else:
            fp_improvement = 0
        
        # 假阴性率 (攻击流量被误判为正常)
        attack_mask = (y_true != 0)
        if np.sum(attack_mask) > 0:
            fn_rate_a = np.sum(pred_a[attack_mask] == 0) / np.sum(attack_mask)
            fn_rate_b = np.sum(pred_b[attack_mask] == 0) / np.sum(attack_mask)
            fn_improvement = (fn_rate_a - fn_rate_b) / fn_rate_a * 100 if fn_rate_a > 0 else 0
        else:
            fn_improvement = 0
        
        # 关键攻击检测率 (DDoS, 暴力破解)
        critical_mask = np.isin(y_true, [AttackType.DDOS.value, AttackType.BRUTE_FORCE.value])
        if np.sum(critical_mask) > 0:
            critical_recall_a = np.sum(pred_a[critical_mask] == y_true[critical_mask]) / np.sum(critical_mask)
            critical_recall_b = np.sum(pred_b[critical_mask] == y_true[critical_mask]) / np.sum(critical_mask)
            critical_improvement = (critical_recall_b - critical_recall_a) / critical_recall_a * 100 if critical_recall_a > 0 else 0
        else:
            critical_improvement = 0
        
        return {
            'false_positive_reduction_pct': fp_improvement,
            'false_negative_reduction_pct': fn_improvement,
            'critical_attack_detection_improvement_pct': critical_improvement,
            'overall_accuracy_improvement_pct': (accuracy_score(y_true, pred_b) - accuracy_score(y_true, pred_a)) / accuracy_score(y_true, pred_a) * 100
        }


def main():
    """主函数 - 风控模型优化与A/B测试演示"""
    print("🚀 风控模型优化系统 - 随机森林增强 + LightGBM + A/B测试")
    print("="*70)
    
    # 1. 生成增强特征数据集
    print("📊 生成增强特征工程数据集...")
    X: np.ndarray
    y: np.ndarray
    X, y = AdvancedFeatureEngineering.generate_balanced_dataset(total_samples=4000)
    # 强制类型转换以解决类型推断问题
    X = cast(np.ndarray, X)
    y = cast(np.ndarray, y)
    
    print(f"数据集规模: {X.shape}")
    print(f"特征维度: {len(AdvancedFeatureEngineering().feature_names)}")
    print(f"类别分布: {dict(zip(['正常', '爬虫', '暴力破解', '订单欺诈', '支付欺诈', 'DDoS'], np.bincount(y)))}")
    
    # 2. 数据预处理和分割
    scaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)
    
    # 三重分割：训练/验证/测试
    temp_result = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )
    X_temp: np.ndarray = cast(np.ndarray, temp_result[0])
    X_test: np.ndarray = cast(np.ndarray, temp_result[1])
    y_temp: np.ndarray = cast(np.ndarray, temp_result[2])
    y_test: np.ndarray = cast(np.ndarray, temp_result[3])
    
    train_result = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    X_train: np.ndarray = cast(np.ndarray, train_result[0])
    X_val: np.ndarray = cast(np.ndarray, train_result[1])
    y_train: np.ndarray = cast(np.ndarray, train_result[2])
    y_val: np.ndarray = cast(np.ndarray, train_result[3])
    
    print(f"\n数据分割 - 训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 3. 训练优化模型
    print("\n🎯 开始训练优化模型...")
    
    # 随机森林 - 基础版
    rf_basic = OptimizedRandomForestModel(optimization_level='basic')
    rf_basic_metrics = rf_basic.train_with_hyperparameter_tuning(X_train, y_train, X_val, y_val)
    
    # 随机森林 - 高级版  
    rf_advanced = OptimizedRandomForestModel(optimization_level='advanced')
    rf_advanced_metrics = rf_advanced.train_with_hyperparameter_tuning(X_train, y_train, X_val, y_val)
    
    # LightGBM模型
    lgb_model = LightGBMModel()
    lgb_metrics = lgb_model.train_optimized(X_train, y_train, X_val, y_val)
    
    # 4. 性能对比分析
    print("\n📈 模型性能对比分析:")
    print("-" * 80)
    
    models_info = [
        ('随机森林-基础版', rf_basic_metrics),
        ('随机森林-高级版', rf_advanced_metrics)
    ]
    
    if lgb_metrics is not None:
        models_info.append(('LightGBM', lgb_metrics))
    
    for model_name, metrics in models_info:
        print(f"{model_name:15} | F1: {metrics.f1_score:.4f} | 准确率: {metrics.accuracy:.4f} | "
              f"AUC: {metrics.auc_score:.4f} | 训练时间: {metrics.training_time:.2f}s")
    
    # 5. 特征重要性分析
    print("\n🔍 特征重要性分析 (Top 10):")
    print("-" * 50)
    
    # 显示高级随机森林的特征重要性
    top_features = sorted(rf_advanced_metrics.feature_importance.items(), 
                         key=lambda x: x[1], reverse=True)[:10]
    
    for i, (feature, importance) in enumerate(top_features, 1):
        print(f"{i:2d}. {feature:25} {importance:.4f}")
    
    # 6. A/B测试统计分析
    print("\n🧪 A/B测试统计分析:")
    print("=" * 50)
    
    ab_framework = BayesianABTestFramework()
    
    # 在测试集上获取预测结果
    test_predictions = {}
    available_models = []
    
    if rf_basic.model is not None:
        test_predictions['RF-基础版'] = rf_basic.model.predict(X_test)
        available_models.append(('RF-基础版', rf_basic.model))
    
    if rf_advanced.model is not None:
        test_predictions['RF-高级版'] = rf_advanced.model.predict(X_test)
        available_models.append(('RF-高级版', rf_advanced.model))
    
    if lgb_model.model is not None:
        test_predictions['LightGBM'] = lgb_model.model.predict(X_test)
        available_models.append(('LightGBM', lgb_model.model))
    
    # 执行两两A/B测试
    ab_results = []
    model_names = list(test_predictions.keys())
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model_a, model_b = model_names[i], model_names[j]
            
            result = ab_framework.mcnemar_test_with_business_metrics(
                model_a, model_b,
                test_predictions[model_a],
                test_predictions[model_b],
                y_test
            )
            ab_results.append(result)
            
            # 输出A/B测试结果
            print(f"\n{model_a} vs {model_b}:")
            print(f"  样本大小: {result.sample_size}")
            print(f"  p-value: {result.p_value:.6f}")
            print(f"  效应大小: {result.effect_size:.4f}")
            print(f"  统计显著性: {'是' if result.is_significant else '否'}")
            print(f"  获胜者: {result.winner}")
            print(f"  置信区间: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            
            # 业务影响分析
            print(f"  业务影响:")
            for metric, value in result.business_impact.items():
                print(f"    {metric}: {value:.2f}%")
    
    # 7. 最终结论和建议
    print("\n🏆 最终结论和算法理论分析:")
    print("=" * 70)
    
    # 找出最佳模型
    best_model_info = max(models_info, key=lambda x: x[1].f1_score)
    best_model_name, best_metrics = best_model_info
    
    print(f"\n推荐最佳模型: {best_model_name}")
    print(f"性能指标: F1={best_metrics.f1_score:.4f}, AUC={best_metrics.auc_score:.4f}")
    
    # 统计显著性总结
    significant_results = [r for r in ab_results if r.is_significant]
    print(f"\n统计显著性结果: {len(significant_results)}/{len(ab_results)} 组对比具有显著差异")
    
    # 算法理论总结
    print("\n📚 算法理论与优化策略总结:")
    print("\n1. 随机森林优化理论:")
    print("   • 基础版 vs 高级版: 通过超参数网格搜索优化")
    print("   • 特征工程: 引入高级组合特征和交互项")
    print("   • 类别均衡: 使用balanced权重处理不平衡数据")
    
    if LIGHTGBM_AVAILABLE:
        print("\n2. LightGBM vs 随机森林对比:")
        print("   • LightGBM: 梯度提升，基于直方图优化，叶子生长策略")
        print("   • 随机森林: 并行bagging，减少方差，可解释性强")
        print("   • 选择建议: 根据数据规模和解释性需求决定")
    
    print("\n3. A/B测试理论框架:")
    print("   • 麦克内马检验: 适用于配对样本的二分类结果比较")
    print("   • 业务指标: 关注假阳性/假阴性对业务的实际影响")
    print("   • 置信区间: 提供准确率差异的不确定性量化")
    
    print("\n4. 特征工程优化:")
    print("   • 基础KPI + 比率特征 + 时间特征 + 熵值特征")
    print("   • 高级组合: 风险-支付交互、时间风险分数、行为一致性")
    print("   • 领域知识: 基于风控业务场景的特征构造")
    
    print("\n✅ 风控模型优化完成! 建议根据A/B测试结果选择最优模型部署。")


if __name__ == "__main__":
    main()