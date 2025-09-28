import os

# 测试恶意链接检测代码
import sys
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

try:
    import pandas as pd
    import numpy as np
    import tldextract
    import re
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from collections import Counter
    import time
    print("✅ 所有依赖库导入成功")
except ImportError as e:
    print(f"❌ 导入库失败: {e}")
    sys.exit(1)

# 模拟测试数据
print("\n🔄 创建测试数据...")
data = {
    'user_id': [1, 1, 1, 1, 2],
    'timestamp': ['2025-09-25 12:00:00', '2025-09-25 12:05:00', '2025-09-25 12:10:00', '2025-09-25 12:15:00', '2025-09-25 12:00:00'],
    'url': ['http://example.com/offer?id=1', 
            'http://malicious.com/bad-link', 
            'http://example.com/offer?id=2', 
            'http://malicious.com/phishing', 
            'http://example.com/offer?id=3']
}

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"✅ 测试数据创建成功，共 {len(df)} 条记录")

# 简化的URL特征提取
def extract_url_features_simple(url):
    try:
        # 基础特征
        url_length = len(url)
        has_https = 1 if url.startswith('https') else 0
        
        # 简单关键词检测
        suspicious_keywords = ['bad', 'phishing', 'malicious']
        keyword_score = sum([1 for word in suspicious_keywords if word in url.lower()])
        
        return {
            'url_length': url_length,
            'has_https': has_https,
            'keyword_score': keyword_score,
        }
    except Exception as e:
        print(f"⚠️ URL解析错误: {url}, 错误: {e}")
        return {'url_length': 0, 'has_https': 0, 'keyword_score': 0}

print("\n🔄 提取URL特征...")
features = df['url'].apply(extract_url_features_simple)
features_df = pd.DataFrame(features.tolist())
df = pd.concat([df, features_df], axis=1)
print("✅ URL特征提取完成")

# 简化的风险评分
def calculate_simple_risk_score(row):
    score = 0
    score += row['url_length'] * 0.01
    score += row['keyword_score'] * 3.0
    score -= row['has_https'] * 1.0
    return max(0, score)

print("\n🔄 计算风险评分...")
df['risk_score'] = df.apply(calculate_simple_risk_score, axis=1)
print("✅ 风险评分计算完成")

# 异常检测
print("\n🔄 执行异常检测...")
try:
    feature_columns = ['url_length', 'keyword_score', 'has_https']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])
    
    model = IsolationForest(contamination='auto', random_state=42)
    anomaly_labels = model.fit_predict(scaled_features)
    anomaly_scores = model.decision_function(scaled_features)
    
    df['anomaly'] = anomaly_labels
    df['anomaly_score'] = anomaly_scores
    print("✅ 异常检测完成")
except Exception as e:
    print(f"❌ 异常检测失败: {e}")
    sys.exit(1)

# 显示结果
print("\n📊 检测结果:")
print("=" * 60)
for idx, row in df.iterrows():
    status = "🚨 恶意" if row['anomaly'] == -1 else "✅ 正常"
    print(f"{status} | 用户{row['user_id']} | 风险评分: {row['risk_score']:.2f}")
    print(f"      URL: {row['url']}")
    print(f"      异常评分: {row['anomaly_score']:.3f}")
    print("-" * 60)

print("\n🎉 测试完成！代码运行正常。")