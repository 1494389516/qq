#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东手机商品爬虫示例
"""

import os
import sys
import time
import logging
import random
import argparse
from config import SPIDER_CONFIG
from jd_main import run_spider, update_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 京东手机商品URL列表
PHONE_URLS = [
    "https://item.jd.com/100049398992.html",  # 苹果 iPhone 15 Pro
    "https://item.jd.com/100050401004.html",  # 华为 Mate 60 Pro
    "https://item.jd.com/100055967220.html",  # 小米 14 Pro
    "https://item.jd.com/100044981190.html",  # 三星 Galaxy S23 Ultra
    "https://item.jd.com/100037289178.html",  # OPPO Find X6 Pro
]

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='京东手机商品爬虫')
    
    parser.add_argument('-o', '--output', type=str,
                        default="results/phones",
                        help='结果保存目录')
    
    parser.add_argument('--headless', action='store_true',
                        default=False,
                        help='是否使用无头浏览器模式')
    
    parser.add_argument('--login', action='store_true',
                        default=False,
                        help='是否需要登录')
    
    parser.add_argument('--username', type=str,
                        default=None,
                        help='京东账号（用于登录）')
    
    parser.add_argument('--password', type=str,
                        default=None,
                        help='京东密码（用于登录）')
    
    return parser.parse_args()

def crawl_phones():
    """爬取多个手机商品信息"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 更新配置
    class Args:
        pass
    
    config_args = Args()
    config_args.output = args.output
    config_args.headless = args.headless
    config_args.login = args.login
    config_args.username = args.username
    config_args.password = args.password
    
    # 创建输出目录
    if not os.path.exists(config_args.output):
        os.makedirs(config_args.output)
    
    # 显示爬取信息
    logger.info("=" * 50)
    logger.info("京东手机商品爬虫启动")
    logger.info(f"输出目录: {config_args.output}")
    logger.info(f"无头模式: {'是' if config_args.headless else '否'}")
    logger.info(f"启用登录: {'是' if config_args.login else '否'}")
    if config_args.login and config_args.username:
        logger.info(f"登录账号: {config_args.username}")
    logger.info("=" * 50)
    
    # 遍历爬取每个手机商品
    for i, url in enumerate(PHONE_URLS):
        try:
            logger.info("=" * 50)
            logger.info(f"正在爬取第 {i+1}/{len(PHONE_URLS)} 个商品: {url}")
            
            # 更新目标URL
            config_args.url = url
            update_config(config_args)
            
            # 运行爬虫
            run_spider(url)
            
            # 随机等待一段时间，避免被反爬
            if i < len(PHONE_URLS) - 1:
                wait_time = random.uniform(3, 6)
                logger.info(f"等待 {wait_time:.2f} 秒后继续...")
                time.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"爬取商品失败: {url}, 错误: {e}")
            continue
    
    logger.info("=" * 50)
    logger.info(f"所有手机商品爬取完成，结果保存在: {config_args.output}")

if __name__ == "__main__":
    try:
        crawl_phones()
    except KeyboardInterrupt:
        logger.info("用户中断爬虫")
        sys.exit(1)
    except Exception as e:
        logger.error(f"爬虫运行出错: {e}")
        sys.exit(1)
    sys.exit(0) 