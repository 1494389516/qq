#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东商城爬虫配置文件
"""

import os
from datetime import datetime

# 爬虫配置
SPIDER_CONFIG = {
    # 目标URL
    'TARGET_URL': 'https://item.jd.com/100050401004.html',  # 华为Mate60 Pro
    
    # 浏览器配置
    'BROWSER': {
        'HEADLESS': False,  # 是否使用无头模式
        'WINDOW_SIZE': (1920, 1080),  # 窗口大小
        'PAGE_LOAD_TIMEOUT': 30,  # 页面加载超时时间（秒）
        'IMPLICIT_WAIT': 10,  # 隐式等待时间（秒）
        'STEALTH_MODE': True,  # 是否使用隐身模式
        'USE_UNDETECTED_CHROMEDRIVER': True,  # 是否使用undetected_chromedriver
    },
    
    # 登录配置
    'LOGIN': {
        'ENABLE': True,  # 是否启用登录，默认开启
        'USERNAME': '',  # 京东账号
        'PASSWORD': '',  # 京东密码
        'COOKIES_DIR': 'cookies',  # cookies保存目录
        'QRCODE_TIMEOUT': 60,  # 二维码超时时间（秒）
        'QRCODE_REFRESH': True,  # 是否自动刷新二维码
        'MAX_QRCODE_REFRESH': 3,  # 最大二维码刷新次数
        'LOGIN_TIMEOUT': 180,  # 登录超时时间（秒）
        'AUTO_OPEN_QRCODE': True,  # 是否自动打开二维码图片
        'SAVE_QRCODE': True,  # 是否保存二维码图片
    },
    
    # 输出配置
    'OUTPUT': {
        'RESULTS_DIR': os.path.join('results', 'huawei'),  # 结果保存目录
        'SAVE_IMAGES': False,  # 是否保存商品图片
        'DEBUG_DIR': os.path.join('results', 'debug'),  # 调试信息保存目录
        'LOG_DIR': 'logs',  # 日志保存目录
        'LOG_LEVEL': 'INFO',  # 日志级别
    },
    
    # 爬虫行为配置
    'BEHAVIOR': {
        'RANDOM_DELAY': True,  # 是否使用随机延迟
        'MIN_DELAY': 1,  # 最小延迟时间（秒）
        'MAX_DELAY': 3,  # 最大延迟时间（秒）
        'SIMULATE_HUMAN': True,  # 是否模拟人类行为
        'MAX_RETRIES': 3,  # 最大重试次数
        'RETRY_DELAY': 5,  # 重试延迟时间（秒）
    },
    
    # 验证码处理配置
    'VERIFICATION': {
        'AUTO_HANDLE': True,  # 是否自动处理验证码
        'SAVE_SCREENSHOTS': True,  # 是否保存验证码截图
        'SCREENSHOTS_DIR': os.path.join('results', 'verification'),  # 验证码截图保存目录
        'MAX_ATTEMPTS': 3,  # 最大尝试次数
    },
    
    # URL重定向处理配置
    'URL_REDIRECT': {
        'HANDLE_REDIRECT': True,  # 是否处理URL重定向
        'MAX_REDIRECTS': 5,  # 最大重定向次数
        'RESTORE_ORIGINAL': True,  # 是否尝试恢复原始URL
        'EXTRACT_PRODUCT_ID': True,  # 是否从URL中提取商品ID
        'FORCE_PC': True,  # 是否强制使用PC版页面（防止重定向到移动版）
        'REJECT_M_PAGES': True,  # 是否拒绝移动版页面
        'DIRECT_PC_DOMAIN': True,  # 是否直接使用PC端域名
        'USER_AGENT_ROTATION': True,  # 是否轮换User-Agent
    },
    
    # 代理配置
    'PROXY': {
        'ENABLE': False,  # 是否使用代理
        'TYPE': 'http',  # 代理类型
        'HOST': '',  # 代理主机
        'PORT': '',  # 代理端口
        'USERNAME': '',  # 代理用户名
        'PASSWORD': '',  # 代理密码
    },
    
    # 调试配置
    'DEBUG': {
        'ENABLE': True,  # 是否启用调试模式
        'SAVE_HTML': True,  # 是否保存HTML
        'SAVE_SCREENSHOTS': True,  # 是否保存截图
        'VERBOSE_LOGGING': True,  # 是否启用详细日志
    }
}

# 创建必要的目录
for dir_path in [
    SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'],
    SPIDER_CONFIG['OUTPUT']['DEBUG_DIR'],
    SPIDER_CONFIG['OUTPUT']['LOG_DIR'],
    SPIDER_CONFIG['LOGIN']['COOKIES_DIR'],
    SPIDER_CONFIG['VERIFICATION']['SCREENSHOTS_DIR']
]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

# Scrapy设置
SCRAPY_SETTINGS = {
    'BOT_NAME': 'jd_spider',
    'ROBOTSTXT_OBEY': False,
    'COOKIES_ENABLED': True,
    'DOWNLOAD_DELAY': SPIDER_CONFIG['BEHAVIOR']['RETRY_DELAY'],
    'RANDOMIZE_DOWNLOAD_DELAY': SPIDER_CONFIG['BEHAVIOR']['RANDOM_DELAY'],
    'USER_AGENT': SPIDER_CONFIG['BROWSER']['USE_UNDETECTED_CHROMEDRIVER'],
    'DOWNLOADER_MIDDLEWARES': {
        '京东商城.JDSeleniumMiddleware': 543,
    },
    'ITEM_PIPELINES': {
        'jd_pipeline.JDProductPipeline': 300,
    },
    'FEED_EXPORT_ENCODING': 'utf-8',
    'LOG_LEVEL': 'INFO',
    'RETRY_TIMES': SPIDER_CONFIG['BEHAVIOR']['MAX_RETRIES'],
    'RETRY_HTTP_CODES': [500, 502, 503, 504, 522, 524, 408, 429],
} 