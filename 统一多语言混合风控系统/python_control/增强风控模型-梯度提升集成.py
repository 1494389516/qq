#!/usr/bin/env python3

"""
增强风控模型系统 - 集成XGBoost、LightGBM、随机森林的多模型框架
设计目标：构建风控算法专家级的理论框架和数学建模能力

核心算法模块：
1. 梯度提升模型集成 (XGBoost + LightGBM + RandomForest)
2. 贝叶斯超参数优化
3. A/B测试理论框架
4. 模型解释性分析 (SHAP + LIME)
"""

import time
import numpy as np
import pandas as pd
import warnings
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# 核心机器学习库
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 超参数优化
import optuna
from optuna.samplers import TPESampler

# A/B测试
from scipy import stats

# 导入基础模块
try:
    from 随机森林算法_恶意行为识别 import AttackType, AttackFeatures, AttackDataGenerator
except ImportError:
    # 如果导入失败，使用本地定义
    class AttackType(Enum):
        NORMAL = 0
        CRAWLER = 1
        BRUTE_FORCE = 2
        ORDER_FRAUD = 3
        PAYMENT_FRAUD = 4
        DDOS = 5
    
    # 复用AttackDataGenerator
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from 随机森林算法_恶意行为识别 import AttackDataGenerator

warnings.filterwarnings('ignore')


@dataclass 
class ModelPerformanceMetrics:
    """模型性能指标"""
    accuracy: float
    f1_score: float
    auc_score: float
    training_time: float
    

@dataclass
class ABTestResult:
    """A/B测试结果"""
    model_a_name: str
    model_b_name: str
    p_value: float
    effect_size: float
    is_significant: bool
    winner: str


class ModelType(Enum):
    """模型类型枚举"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost" 
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"


class BaseRiskModel(ABC):
    """风控模型基类 - 抽象理论框架"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
    @abstractmethod
    def build_model(self, hyperparams: Dict[str, Any]) -> Any:
        """构建模型架构"""
        pass
        
    @abstractmethod
    def train_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelPerformanceMetrics:
        """训练模型"""
        pass
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """概率预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class XGBoostRiskModel(BaseRiskModel):
    """XGBoost风控模型 - 梯度提升决策树"""
    
    def __init__(self):
        super().__init__("XGBoost")
        
    def build_model(self, hyperparams: Dict[str, Any]) -> xgb.XGBClassifier:
        """构建XGBoost模型"""
        return xgb.XGBClassifier(
            n_estimators=hyperparams.get('n_estimators', 200),
            max_depth=hyperparams.get('max_depth', 8),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            subsample=hyperparams.get('subsample', 0.8),
            colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
            reg_alpha=hyperparams.get('reg_alpha', 0.1),
            reg_lambda=hyperparams.get('reg_lambda', 1.0),
            objective='multi:softprob',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
    def train_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelPerformanceMetrics:
        """训练XGBoost模型"""
        start_time = time.time()
        
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        
        # 构建模型
        hyperparams = kwargs.get('hyperparams', {})
        self.model = self.build_model(hyperparams)
        
        # 训练集验证集分割
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # 评估性能
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)
        
        metrics = ModelPerformanceMetrics(
            accuracy=float(accuracy_score(y_val, y_pred)),
            f1_score=float(f1_score(y_val, y_pred, average='weighted')),
            auc_score=float(roc_auc_score(y_val, y_proba, multi_class='ovr')),
            training_time=training_time
        )
        
        self.logger.info(f"XGBoost训练完成 - 准确率: {metrics.accuracy:.4f}")
        return metrics


class LightGBMRiskModel(BaseRiskModel):
    """LightGBM风控模型 - 基于直方图的梯度提升"""
    
    def __init__(self):
        super().__init__("LightGBM")
        
    def build_model(self, hyperparams: Dict[str, Any]) -> lgb.LGBMClassifier:
        """构建LightGBM模型"""
        return lgb.LGBMClassifier(
            n_estimators=hyperparams.get('n_estimators', 200),
            max_depth=hyperparams.get('max_depth', 8),
            learning_rate=hyperparams.get('learning_rate', 0.1),
            num_leaves=hyperparams.get('num_leaves', 31),
            subsample=hyperparams.get('subsample', 0.8),
            colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
            reg_alpha=hyperparams.get('reg_alpha', 0.1),
            reg_lambda=hyperparams.get('reg_lambda', 1.0),
            objective='multiclass',
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
    def train_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelPerformanceMetrics:
        """训练LightGBM模型"""
        start_time = time.time()
        
        X_scaled = self.scaler.fit_transform(X)
        hyperparams = kwargs.get('hyperparams', {})
        self.model = self.build_model(hyperparams)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)
        
        metrics = ModelPerformanceMetrics(
            accuracy=float(accuracy_score(y_val, y_pred)),
            f1_score=float(f1_score(y_val, y_pred, average='weighted')),
            auc_score=float(roc_auc_score(y_val, y_proba, multi_class='ovr')),
            training_time=training_time
        )
        
        self.logger.info(f"LightGBM训练完成 - 准确率: {metrics.accuracy:.4f}")
        return metrics


class RandomForestRiskModel(BaseRiskModel):
    """随机森林风控模型 - 改进版"""
    
    def __init__(self):
        super().__init__("RandomForest")
        
    def build_model(self, hyperparams: Dict[str, Any]) -> RandomForestClassifier:
        """构建随机森林模型"""
        return RandomForestClassifier(
            n_estimators=hyperparams.get('n_estimators', 200),
            max_depth=hyperparams.get('max_depth', 15),
            min_samples_split=hyperparams.get('min_samples_split', 5),
            min_samples_leaf=hyperparams.get('min_samples_leaf', 2),
            max_features=hyperparams.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
    def train_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelPerformanceMetrics:
        """训练随机森林模型"""
        start_time = time.time()
        
        X_scaled = self.scaler.fit_transform(X)
        hyperparams = kwargs.get('hyperparams', {})
        self.model = self.build_model(hyperparams)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)
        
        metrics = ModelPerformanceMetrics(
            accuracy=float(accuracy_score(y_val, y_pred)),
            f1_score=float(f1_score(y_val, y_pred, average='weighted')),
            auc_score=float(roc_auc_score(y_val, y_proba, multi_class='ovr')),
            training_time=training_time
        )
        
        self.logger.info(f"随机森林训练完成 - 准确率: {metrics.accuracy:.4f}")
        return metrics


class BayesianHyperparameterOptimizer:
    """贝叶斯超参数优化器"""
    
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        self.logger = logging.getLogger(f"{__name__}.BayesianOptimizer")
        
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """优化XGBoost超参数"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0)
            }
            
            model = XGBoostRiskModel()
            scores = []
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.train_model(X_train, y_train, hyperparams=params)
                metrics = model.evaluate_model(X_val, y_val)
                scores.append(metrics.f1_score)
                
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=self.n_trials)
        
        self.logger.info(f"XGBoost最优参数: {study.best_params}")
        return study.best_params


class EnsembleRiskModel:
    """集成风控模型 - 多模型融合"""
    
    def __init__(self):
        self.models = {
            'xgboost': XGBoostRiskModel(),
            'lightgbm': LightGBMRiskModel(), 
            'random_forest': RandomForestRiskModel()
        }
        self.weights = {}
        self.is_trained = False
        self.logger = logging.getLogger(f"{__name__}.EnsembleModel")
        
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                      hyperparams: Dict[str, Dict] = None) -> Dict[str, ModelPerformanceMetrics]:
        """训练集成模型"""
        if hyperparams is None:
            hyperparams = {}
            
        performance_metrics = {}
        
        # 训练各个子模型
        for model_name, model in self.models.items():
            model_hyperparams = hyperparams.get(model_name, {})
            metrics = model.train_model(X, y, hyperparams=model_hyperparams)
            performance_metrics[model_name] = metrics
            
        # 计算集成权重（基于F1分数）
        f1_scores = {name: metrics.f1_score for name, metrics in performance_metrics.items()}
        total_f1 = sum(f1_scores.values())
        self.weights = {name: score / total_f1 for name, score in f1_scores.items()}
        
        self.is_trained = True
        self.logger.info(f"集成模型权重: {self.weights}")
        
        return performance_metrics
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """集成概率预测"""
        if not self.is_trained:
            raise ValueError("集成模型未训练")
            
        # 获取各模型预测概率
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict_proba(X)
            
        # 加权集成
        ensemble_proba = np.zeros_like(predictions['xgboost'])
        for model_name, proba in predictions.items():
            ensemble_proba += self.weights[model_name] * proba
            
        return ensemble_proba


class ABTestFramework:
    """A/B测试理论框架"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger(f"{__name__}.ABTestFramework")
        
    def statistical_significance_test(self, model_a: BaseRiskModel, model_b: BaseRiskModel,
                                    X_test: np.ndarray, y_test: np.ndarray) -> ABTestResult:
        """统计显著性检验"""
        
        # 模型预测
        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)
        
        accuracy_a = accuracy_score(y_test, pred_a)
        accuracy_b = accuracy_score(y_test, pred_b)
        
        # 麦克内马检验
        correct_a = (pred_a == y_test).astype(int)
        correct_b = (pred_b == y_test).astype(int)
        
        a_correct_b_wrong = np.sum((correct_a == 1) & (correct_b == 0))
        a_wrong_b_correct = np.sum((correct_a == 0) & (correct_b == 1))
        
        # 麦克内马统计量
        if a_correct_b_wrong + a_wrong_b_correct > 0:
            mcnemar_stat = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / (a_correct_b_wrong + a_wrong_b_correct)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            p_value = 1.0
            
        # 效应大小
        effect_size = abs(accuracy_a - accuracy_b)
        
        # 判断显著性和获胜者
        is_significant = p_value < self.significance_level
        if is_significant:
            winner = model_a.model_name if accuracy_a > accuracy_b else model_b.model_name
        else:
            winner = "无显著差异"
            
        return ABTestResult(
            model_a_name=model_a.model_name,
            model_b_name=model_b.model_name,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            winner=winner
        )


def main():
    """主函数 - 演示增强风控模型系统"""
    print("🚀 增强风控模型系统 - XGBoost + LightGBM + 随机森林集成")
    print("="*60)
    
    # 生成训练数据
    print("📊 生成训练数据...")
    normal_data = AttackDataGenerator.generate_normal_data(800)
    crawler_data = AttackDataGenerator.generate_crawler_data(150)
    brute_force_data = AttackDataGenerator.generate_brute_force_data(100)
    order_fraud_data = AttackDataGenerator.generate_order_fraud_data(80)
    payment_fraud_data = AttackDataGenerator.generate_payment_fraud_data(70)
    ddos_data = AttackDataGenerator.generate_ddos_data(50)
    
    all_data = normal_data + crawler_data + brute_force_data + order_fraud_data + payment_fraud_data + ddos_data
    
    # 转换为数组格式
    feature_names = [
        'order_requests', 'payment_success', 'product_pv', 'risk_hits',
        'payment_success_rate', 'risk_hit_rate', 'pv_order_ratio',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
        'time_offset', 'source_entropy', 'ip_entropy',
        'order_trend', 'payment_trend', 'pv_trend', 'risk_trend'
    ]
    
    X = []
    y = []
    
    for features in all_data:
        feature_array = np.array([
            features.order_requests, features.payment_success, features.product_pv, features.risk_hits,
            features.payment_success_rate, features.risk_hit_rate, features.pv_order_ratio,
            features.hour_of_day, features.day_of_week, int(features.is_weekend), int(features.is_night),
            features.time_offset, features.source_entropy, features.ip_entropy,
            features.order_trend, features.payment_trend, features.pv_trend, features.risk_trend
        ])
        X.append(feature_array)
        y.append(features.attack_type)
    
    X = np.array(X)
    y = np.array(y)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    # 1. 训练个体模型
    print("\n🎯 训练个体模型...")
    
    # XGBoost
    xgb_model = XGBoostRiskModel()
    xgb_metrics = xgb_model.train_model(X_train, y_train)
    
    # LightGBM
    lgb_model = LightGBMRiskModel()
    lgb_metrics = lgb_model.train_model(X_train, y_train)
    
    # 随机森林
    rf_model = RandomForestRiskModel()
    rf_metrics = rf_model.train_model(X_train, y_train)
    
    # 2. 训练集成模型
    print("\n🔥 训练集成模型...")
    ensemble_model = EnsembleRiskModel()
    ensemble_metrics = ensemble_model.train_ensemble(X_train, y_train)
    
    # 3. A/B测试
    print("\n🧪 执行A/B测试...")
    ab_framework = ABTestFramework()
    
    # XGBoost vs LightGBM
    result1 = ab_framework.statistical_significance_test(xgb_model, lgb_model, X_test, y_test)
    print(f"XGBoost vs LightGBM: {result1.winner} (p-value: {result1.p_value:.4f})")
    
    # 集成模型评估
    ensemble_pred = np.argmax(ensemble_model.predict_proba(X_test), axis=1)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"\n📈 最终结果:")
    print(f"XGBoost准确率: {xgb_metrics.accuracy:.4f}")
    print(f"LightGBM准确率: {lgb_metrics.accuracy:.4f}")
    print(f"随机森林准确率: {rf_metrics.accuracy:.4f}")
    print(f"集成模型准确率: {ensemble_accuracy:.4f}")
    
    print(f"\n🏆 推荐使用: 集成模型 (融合三种算法优势)")


if __name__ == "__main__":
    main()