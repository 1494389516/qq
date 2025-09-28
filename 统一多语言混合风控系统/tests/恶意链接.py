import pandas as pd
import numpy as np
import tldextract
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import Counter
import time

# 模拟交易数据
data = {
    'user_id': [1, 1, 1, 1, 2],
    'timestamp': ['2025-09-25 12:00:00', '2025-09-25 12:05:00', '2025-09-25 12:10:00', '2025-09-25 12:15:00', '2025-09-25 12:00:00'],
    'url': ['http://example.com/offer?id=1', 
            'http://malicious.com/bad-link', 
            'http://example.com/offer?id=2', 
            'http://malicious.com/phishing', 
            'http://example.com/offer?id=3']
}

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 转换 timestamp 为 datetime 类型
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 优化后的 URL 特征提取函数
def extract_url_features(url):
    from urllib.parse import urlparse
    
    # 提取域名和路径
    try:
        ext = tldextract.extract(url)
        parsed_url = urlparse(url)
    except Exception as e:
        print(f"URL解析错误: {url}, 错误: {e}")
        return get_default_features()
    
    domain = ext.domain + '.' + ext.suffix if ext.suffix else ext.domain
    path = parsed_url.path
    query = parsed_url.query
    
    # 基础URL特征
    url_length = len(url)
    path_length = len(path)
    query_length = len(query)
    
    # URL结构分析特征
    is_short_url = 1 if url_length < 30 else 0
    is_long_url = 1 if url_length > 100 else 0
    has_query_param = 1 if query else 0
    has_subdomain = 1 if ext.subdomain else 0
    has_https = 1 if url.startswith('https') else 0
    has_ip_address = 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0
    
    # 可疑字符统计
    special_char_count = len(re.findall(r'[\-\_\%\@\&\=\?]', url))
    digit_count = len(re.findall(r'\d', url))
    
    # 恶意关键词检测（扩展关键词列表）
    suspicious_keywords = [
        'offer', 'bad', 'phishing', 'free', 'click', 'win', 'secure',
        'verify', 'update', 'suspend', 'confirm', 'urgent', 'prize',
        'bonus', 'reward', 'download', 'install', 'activate'
    ]
    keyword_score = sum([1 for word in suspicious_keywords if re.search(rf'\b{word}\b', url.lower())])
    
    # 域名可疑性检测
    domain_suspicious = 0
    if domain:
        # 检查域名是否包含数字
        if re.search(r'\d', domain):
            domain_suspicious += 1
        # 检查域名长度
        if len(domain) > 20:
            domain_suspicious += 1
        # 检查是否为已知可疑域名
        suspicious_domains = ['malicious.com', 'phishing.net', 'scam.org']
        if domain.lower() in suspicious_domains:
            domain_suspicious += 3
    
    return {
        'domain': domain,
        'url_length': url_length,
        'path_length': path_length,
        'query_length': query_length,
        'is_short_url': is_short_url,
        'is_long_url': is_long_url,
        'has_query_param': has_query_param,
        'has_subdomain': has_subdomain,
        'has_https': has_https,
        'has_ip_address': has_ip_address,
        'special_char_count': special_char_count,
        'digit_count': digit_count,
        'keyword_score': keyword_score,
        'domain_suspicious': domain_suspicious,
    }

def get_default_features():
    """返回默认特征值（用于URL解析失败的情况）"""
    return {
        'domain': 'unknown',
        'url_length': 0,
        'path_length': 0,
        'query_length': 0,
        'is_short_url': 0,
        'is_long_url': 0,
        'has_query_param': 0,
        'has_subdomain': 0,
        'has_https': 0,
        'has_ip_address': 0,
        'special_char_count': 0,
        'digit_count': 0,
        'keyword_score': 0,
        'domain_suspicious': 0,
    }

# 提取每个URL的特征
features = df['url'].apply(extract_url_features)
features_df = pd.DataFrame(features.tolist())

# 合并特征和原数据
df = pd.concat([df, features_df], axis=1)

# 优化后的危险性评分算法
def calculate_risk_score(row):
    """基于多维度特征计算URL危险性评分"""
    score = 0
    
    # URL长度相关评分
    score += row['url_length'] * 0.01  # 长URL可能更可疑
    score += row['path_length'] * 0.1
    score += row['query_length'] * 0.15
    
    # 结构特征评分
    score += row['is_short_url'] * 2.0    # 短链接风险较高
    score += row['is_long_url'] * 1.0     # 过长URL也可疑
    score += row['has_ip_address'] * 3.0  # 直接使用IP地址很可疑
    score += row['special_char_count'] * 0.2  # 特殊字符过多可疑
    score += row['digit_count'] * 0.1     # 数字过多可疑
    
    # 内容特征评分
    score += row['keyword_score'] * 2.5   # 恶意关键词权重提高
    score += row['domain_suspicious'] * 2.0  # 可疑域名评分
    
    # 安全特征减分
    score -= row['has_https'] * 1.5       # HTTPS相对安全
    
    # 确保评分为正数
    return max(0, score)

# 计算危险性分数
df['risk_score'] = df.apply(calculate_risk_score, axis=1)

# 使用改进的异常检测模型
def detect_anomalies(df):
    """使用多种特征进行异常检测"""
    # 选择更多特征进行异常检测
    feature_columns = [
        'url_length', 'path_length', 'query_length', 'keyword_score', 
        'is_short_url', 'is_long_url', 'has_ip_address', 
        'special_char_count', 'domain_suspicious'
    ]
    
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])
    
    # 使用Isolation Forest进行异常检测
    model = IsolationForest(
        contamination='auto',   # 自动确定污染率
        random_state=42,        # 设置随机种子确保结果可重现
        n_estimators=200        # 增加树的数量提高准确性
    )
    
    # 先训练模型，再获取异常标签和评分
    anomaly_labels = model.fit_predict(scaled_features)
    anomaly_scores = model.decision_function(scaled_features)
    
    return anomaly_labels, anomaly_scores

# 执行异常检测
df['anomaly'], df['anomaly_score'] = detect_anomalies(df)

# 优化后的用户行为分析
def analyze_user_behavior(df, time_window_minutes=10, suspicious_threshold=2):
    """分析用户点击行为模式"""
    time_window = pd.Timedelta(f'{time_window_minutes} min')
    df['time_diff'] = df.groupby('user_id')['timestamp'].diff().fillna(pd.Timedelta(seconds=0))
    df['frequent_malicious_click'] = 0
    df['click_velocity'] = 0.0  # 点击速度
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('timestamp')
        
        for i, row in user_data.iterrows():
            current_time = row['timestamp']
            time_limit = current_time - time_window
            
            # 计算时间窗口内的点击统计
            window_data = user_data[
                (user_data['timestamp'] >= time_limit) & 
                (user_data['timestamp'] <= current_time)
            ]
            
            # 恶意链接频繁点击检测
            malicious_clicks_in_window = window_data[window_data['anomaly'] == -1]
            if len(malicious_clicks_in_window) > suspicious_threshold:
                df.at[i, 'frequent_malicious_click'] = 1
            
            # 计算点击速度（每分钟点击次数）
            if len(window_data) > 1:
                click_velocity = len(window_data) / time_window_minutes
                df.at[i, 'click_velocity'] = click_velocity
    
    return df

# 执行用户行为分析
df = analyze_user_behavior(df)

# 生成详细的分析报告
def generate_analysis_report(df):
    """生成恶意链接检测分析报告"""
    print("=" * 80)
    print("恶意链接检测分析报告")
    print("=" * 80)
    
    # 基本统计信息
    total_urls = len(df)
    malicious_urls = len(df[df['anomaly'] == -1])
    malicious_rate = (malicious_urls / total_urls) * 100 if total_urls > 0 else 0
    
    print(f"\n📊 总体统计:")
    print(f"   - 总URL数量: {total_urls}")
    print(f"   - 检测到恶意URL: {malicious_urls}")
    print(f"   - 恶意URL比例: {malicious_rate:.2f}%")
    
    # 用户行为统计
    print(f"\n👤 用户行为分析:")
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id]
        user_malicious = len(user_data[user_data['anomaly'] == -1])
        user_frequent = len(user_data[user_data['frequent_malicious_click'] == 1])
        avg_risk_score = user_data['risk_score'].mean()
        print(f"   - 用户 {user_id}: 恶意链接 {user_malicious}个, 频繁点击 {user_frequent}次, 平均风险评分 {avg_risk_score:.2f}")
    
    # 详细结果
    print(f"\n📋 详细检测结果:")
    result_columns = [
        'user_id', 'timestamp', 'url', 'risk_score', 
        'anomaly', 'anomaly_score', 'frequent_malicious_click', 'click_velocity'
    ]
    
    for idx, row in df.iterrows():
        status = "🚨 恶意" if row['anomaly'] == -1 else "✅ 正常"
        print(f"\n   {status} | 用户{row['user_id']} | {row['timestamp']}")
        print(f"      URL: {row['url']}")
        print(f"      风险评分: {row['risk_score']:.2f} | 异常评分: {row['anomaly_score']:.3f}")
        if row['frequent_malicious_click'] == 1:
            print(f"      ⚠️  检测到频繁恶意点击行为")
        if row['click_velocity'] > 0:
            print(f"      点击速度: {row['click_velocity']:.2f} 次/分钟")
    
    print("\n" + "=" * 80)
    return df[result_columns]

# 生成分析报告
result_df = generate_analysis_report(df)

