import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
import hashlib
import uuid

# 模拟IP地址池和设备操作系统
ip_pools = {
    '北京': ['114.247.50.', '123.125.114.', '61.135.169.'],
    '上海': ['101.95.120.', '180.153.214.', '116.228.111.'],
    '广州': ['113.108.230.', '14.215.177.', '119.147.15.'],
    '深圳': ['119.147.15.', '202.104.1.', '58.60.188.'],
    '杭州': ['115.236.112.', '60.12.166.', '101.71.37.'],
    '恶意区域': ['192.168.1.', '10.0.0.', '172.16.0.']  # 可疑IP段
}

os_list = ['Windows 10', 'Windows 11', 'macOS Monterey', 'macOS Ventura', 'iOS 15', 'iOS 16', 'Android 12', 'Android 13', 'Linux Ubuntu']
browsers = ['Chrome', 'Safari', 'Firefox', 'Edge']

# 生成设备指纹
def generate_device_fingerprint(os_type, browser):
    """生成设备指纹"""
    device_id = str(uuid.uuid4())
    screen_resolution = random.choice(['1920x1080', '1366x768', '1440x900', '2560x1440'])
    timezone = random.choice(['GMT+8', 'GMT+0', 'GMT-5'])
    
    fingerprint_string = f"{device_id}_{os_type}_{browser}_{screen_resolution}_{timezone}"
    fingerprint_hash = hashlib.md5(fingerprint_string.encode()).hexdigest()[:16]
    
    return {
        'device_fingerprint': fingerprint_hash,
        'os_type': os_type,
        'browser': browser,
        'screen_resolution': screen_resolution,
        'timezone': timezone
    }

# 生成IP地址
def generate_ip(region, is_malicious=False):
    """生成IP地址"""
    if is_malicious:
        base_ip = random.choice(ip_pools['恶意区域'])
    else:
        base_ip = random.choice(ip_pools[region])
    
    last_segment = random.randint(1, 254)
    return f"{base_ip}{last_segment}"

# 生成测试数据
def generate_test_data():
    """生成20组测试数据：白天10组(1组恶意)，晚上10组(1组恶意)"""
    data_list = []
    user_id = 1
    
    # 白天数据 (8:00-18:00)
    print("生成白天数据...")
    for i in range(10):
        is_malicious = (i == 5)  # 第6组为恶意
        
        if is_malicious:
            # 恶意行为：短时间内大额交易
            region = '恶意区域'
            base_time = datetime(2025, 9, 25, random.randint(8, 18), random.randint(0, 59))
            amounts = [random.randint(500, 2000) for _ in range(3)]  # 大额交易
            time_intervals = [timedelta(seconds=random.randint(10, 30)) for _ in range(2)]  # 短时间间隔
        else:
            # 正常行为
            region = random.choice(['北京', '上海', '广州', '深圳', '杭州'])
            base_time = datetime(2025, 9, 25, random.randint(8, 18), random.randint(0, 59))
            amounts = [random.randint(50, 300) for _ in range(random.randint(1, 3))]
            time_intervals = [timedelta(minutes=random.randint(5, 30)) for _ in range(len(amounts)-1)]
        
        # 生成设备信息
        os_type = random.choice(os_list)
        browser = random.choice(browsers)
        device_info = generate_device_fingerprint(os_type, browser)
        ip_address = generate_ip(region, is_malicious)
        
        # 生成交易记录
        current_time = base_time
        for j, amount in enumerate(amounts):
            data_list.append({
                'user_id': user_id,
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'amount': amount,
                'category': random.choice(['electronics', 'clothing', 'books', 'accessories', 'food']),
                'ip_address': ip_address,
                'region': region,
                'device_fingerprint': device_info['device_fingerprint'],
                'os_type': device_info['os_type'],
                'browser': device_info['browser'],
                'screen_resolution': device_info['screen_resolution'],
                'timezone': device_info['timezone'],
                'is_malicious': is_malicious,
                'time_period': '白天'
            })
            
            if j < len(time_intervals):
                current_time += time_intervals[j]
        
        user_id += 1
    
    # 晚上数据 (19:00-23:59)
    print("生成晚上数据...")
    for i in range(10):
        is_malicious = (i == 7)  # 第8组为恶意
        
        if is_malicious:
            # 恶意行为：异常地区+设备指纹变化
            region = '恶意区域'
            base_time = datetime(2025, 9, 25, random.randint(19, 23), random.randint(0, 59))
            amounts = [random.randint(800, 1500) for _ in range(4)]  # 大额多次交易
            time_intervals = [timedelta(minutes=random.randint(1, 3)) for _ in range(3)]  # 极短时间间隔
        else:
            # 正常行为
            region = random.choice(['北京', '上海', '广州', '深圳', '杭州'])
            base_time = datetime(2025, 9, 25, random.randint(19, 23), random.randint(0, 59))
            amounts = [random.randint(100, 500) for _ in range(random.randint(1, 2))]
            time_intervals = [timedelta(minutes=random.randint(10, 60)) for _ in range(len(amounts)-1)]
        
        # 生成设备信息
        os_type = random.choice(os_list)
        browser = random.choice(browsers)
        device_info = generate_device_fingerprint(os_type, browser)
        ip_address = generate_ip(region, is_malicious)
        
        # 生成交易记录
        current_time = base_time
        for j, amount in enumerate(amounts):
            data_list.append({
                'user_id': user_id,
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'amount': amount,
                'category': random.choice(['electronics', 'clothing', 'books', 'accessories', 'food']),
                'ip_address': ip_address,
                'region': region,
                'device_fingerprint': device_info['device_fingerprint'],
                'os_type': device_info['os_type'],
                'browser': device_info['browser'],
                'screen_resolution': device_info['screen_resolution'],
                'timezone': device_info['timezone'],
                'is_malicious': is_malicious,
                'time_period': '晚上'
            })
            
            if j < len(time_intervals):
                current_time += time_intervals[j]
        
        user_id += 1
    
    return data_list

# 生成测试数据
test_data = generate_test_data()
data = pd.DataFrame(test_data)

# 数据已经是DataFrame格式
df = data

# 转换 timestamp 为 datetime 类型
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 对数据按照时间排序
df = df.sort_values(by=['timestamp']).reset_index(drop=True)

# 高级行为识别系统
class AdvancedBehaviorDetector:
    def __init__(self):
        # 白天监控策略 (8:00-18:59)
        self.day_thresholds = {
            'amount_limit': 500,  # 单笔金额限制
            'frequency_limit': 3,  # 10分钟内交易次数
            'time_window': 10,    # 时间窗口(分钟)
            'total_amount_limit': 1000  # 时间窗口内总金额
        }
        
        # 晚上监控策略 (19:00-07:59)
        self.night_thresholds = {
            'amount_limit': 300,  # 更严格的单笔金额限制
            'frequency_limit': 2,  # 更严格的频次限制
            'time_window': 15,    # 更长的监控窗口
            'total_amount_limit': 600  # 更严格的总金额限制
        }
        
        # 可疑IP段
        self.suspicious_ips = ['192.168.1.', '10.0.0.', '172.16.0.']
    
    def get_time_period(self, timestamp):
        """判断时间段"""
        hour = timestamp.hour
        return 'day' if 8 <= hour <= 18 else 'night'
    
    def is_suspicious_ip(self, ip_address):
        """检查是否为可疑IP"""
        return any(ip_address.startswith(suspicious) for suspicious in self.suspicious_ips)
    
    def detect_frequency_anomaly(self, user_data, current_index):
        """检测频率异常"""
        current_time = user_data.iloc[current_index]['timestamp']
        time_period = self.get_time_period(current_time)
        
        thresholds = self.day_thresholds if time_period == 'day' else self.night_thresholds
        time_window = pd.Timedelta(minutes=thresholds['time_window'])
        
        # 获取时间窗口内的交易
        window_start = current_time - time_window
        # 使用.loc进行布尔索引以避免类型检查警告
        time_mask = (user_data['timestamp'] >= window_start) & (user_data['timestamp'] <= current_time)
        window_transactions = user_data[time_mask]
        
        frequency_score = 0
        if len(window_transactions) > thresholds['frequency_limit']:
            frequency_score += 30
        
        total_amount = window_transactions['amount'].sum()
        if total_amount > thresholds['total_amount_limit']:
            frequency_score += 25
        
        return frequency_score
    
    def detect_amount_anomaly(self, amount, timestamp):
        """检测金额异常"""
        time_period = self.get_time_period(timestamp)
        thresholds = self.day_thresholds if time_period == 'day' else self.night_thresholds
        
        if amount > thresholds['amount_limit']:
            return 20
        return 0
    
    def detect_device_anomaly(self, user_data, current_index):
        """检测设备指纹变化"""
        if current_index == 0:
            return 0
        
        current_fingerprint = user_data.iloc[current_index]['device_fingerprint']
        prev_fingerprint = user_data.iloc[current_index-1]['device_fingerprint']
        
        if current_fingerprint != prev_fingerprint:
            return 15
        return 0
    
    def detect_location_anomaly(self, user_data, current_index):
        """检测地理位置异常"""
        current_ip = user_data.iloc[current_index]['ip_address']
        
        # 检查是否为可疑IP
        if self.is_suspicious_ip(current_ip):
            return 40
        
        # 检查地理位置变化
        if current_index > 0:
            current_region = user_data.iloc[current_index]['region']
            prev_region = user_data.iloc[current_index-1]['region']
            
            if current_region != prev_region:
                return 10
        
        return 0
    
    def calculate_risk_score(self, user_data, current_index):
        """计算风险评分"""
        current_row = user_data.iloc[current_index]
        
        # 各种异常检测
        frequency_score = self.detect_frequency_anomaly(user_data, current_index)
        amount_score = self.detect_amount_anomaly(current_row['amount'], current_row['timestamp'])
        device_score = self.detect_device_anomaly(user_data, current_index)
        location_score = self.detect_location_anomaly(user_data, current_index)
        
        total_score = frequency_score + amount_score + device_score + location_score
        
        return {
            'total_score': total_score,
            'frequency_score': frequency_score,
            'amount_score': amount_score,
            'device_score': device_score,
            'location_score': location_score
        }
    
    def classify_risk_level(self, risk_score, time_period):
        """根据时间段和风险评分分类风险等级"""
        if time_period == 'day':
            # 白天：低监控策略
            if risk_score >= 50:
                return '高风险'
            elif risk_score >= 30:
                return '中风险'
            else:
                return '低风险'
        else:
            # 晚上：强监控策略
            if risk_score >= 35:
                return '高风险'
            elif risk_score >= 20:
                return '中风险'
            else:
                return '低风险'

# 创建检测器实例
detector = AdvancedBehaviorDetector()

# 对每个用户的交易进行风险评估
results = []
for user_id in df['user_id'].unique():
    user_data = df[df['user_id'] == user_id].reset_index(drop=True)
    
    for i in range(len(user_data)):
        risk_scores = detector.calculate_risk_score(user_data, i)
        time_period = detector.get_time_period(user_data.iloc[i]['timestamp'])
        risk_level = detector.classify_risk_level(risk_scores['total_score'], time_period)
        
        result = user_data.iloc[i].copy()
        result['risk_score'] = risk_scores['total_score']
        result['frequency_score'] = risk_scores['frequency_score']
        result['amount_score'] = risk_scores['amount_score']
        result['device_score'] = risk_scores['device_score']
        result['location_score'] = risk_scores['location_score']
        result['risk_level'] = risk_level
        result['monitoring_strategy'] = '低监控' if time_period == 'day' else '强监控'
        
        results.append(result)

# 转换为DataFrame
result_df = pd.DataFrame(results)

# 显示结果统计
print("=== 行为识别系统分析结果 ===")
print(f"\n总交易数量: {len(result_df)}")
print(f"\n时间段分布:")
print(result_df['time_period'].value_counts())

print(f"\n风险等级分布:")
print(result_df['risk_level'].value_counts())

print(f"\n监控策略分布:")
print(result_df['monitoring_strategy'].value_counts())

# 显示高风险交易详情
high_risk_transactions = result_df[result_df['risk_level'] == '高风险']
print(f"\n=== 高风险交易详情 (共{len(high_risk_transactions)}笔) ===")
for _, transaction in high_risk_transactions.iterrows():
    print(f"\n用户ID: {transaction['user_id']}")
    print(f"时间: {transaction['timestamp']} ({transaction['time_period']})")
    print(f"金额: ¥{transaction['amount']}")
    print(f"地区: {transaction['region']} (IP: {transaction['ip_address']})")
    print(f"设备: {transaction['os_type']} - {transaction['browser']}")
    print(f"风险评分: {transaction['risk_score']} (频率:{transaction['frequency_score']}, 金额:{transaction['amount_score']}, 设备:{transaction['device_score']}, 位置:{transaction['location_score']})")
    print(f"监控策略: {transaction['monitoring_strategy']}")
    print(f"实际标签: {'恶意' if bool(transaction['is_malicious']) else '正常'}")
    print("-" * 50)

# 验证检测效果
# 使用布尔掩码避免类型检查警告
malicious_mask = (result_df['is_malicious'] == True) & (result_df['risk_level'] == '高风险')
malicious_detected = len(result_df[malicious_mask])
malicious_total = len(result_df[result_df['is_malicious'] == True])
normal_mask = (result_df['is_malicious'] == False) & (result_df['risk_level'] == '高风险')
normal_misclassified = len(result_df[normal_mask])

print(f"\n=== 检测效果评估 ===")
print(f"恶意交易检出率: {malicious_detected}/{malicious_total} = {malicious_detected/malicious_total*100:.1f}%")
print(f"正常交易误报数: {normal_misclassified}")
print(f"总体准确性评估: {'良好' if malicious_detected/malicious_total >= 0.8 and normal_misclassified <= 2 else '需要调优'}")

# 保存详细结果
print(f"\n=== 详细交易记录 ===")
columns_to_show = ['user_id', 'timestamp', 'amount', 'region', 'ip_address', 'os_type', 
                  'risk_score', 'risk_level', 'monitoring_strategy', 'is_malicious']
print(result_df[columns_to_show].to_string(index=False))

