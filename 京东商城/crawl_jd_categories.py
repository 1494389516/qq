#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东多品类商品爬虫
支持按品类爬取京东商品数据
"""

import os
import sys
import time
import json
import logging
import random
import argparse
from datetime import datetime
from config import SPIDER_CONFIG
from jd_main import run_spider, update_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 检查是否在Windows环境中，以支持彩色输出
try:
    import colorama
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

# 彩色输出配置
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m" if HAS_COLORAMA else ""
    GREEN = "\033[92m" if HAS_COLORAMA else ""
    YELLOW = "\033[93m" if HAS_COLORAMA else ""
    BLUE = "\033[94m" if HAS_COLORAMA else ""
    BOLD = "\033[1m" if HAS_COLORAMA else ""
    CYAN = "\033[96m" if HAS_COLORAMA else ""

# 京东各品类热门商品示例
JD_CATEGORIES = {
    "手机": [
        {"name": "iPhone 15 Pro", "url": "https://item.jd.com/10090748186604.html"},
        {"name": "华为 Mate 60 Pro", "url": "https://item.jd.com/100050401004.html"},
        {"name": "小米 14", "url": "https://item.jd.com/10074798078314.html"},
        {"name": "OPPO Find X7", "url": "https://item.jd.com/10083867113659.html"},
        {"name": "vivo X100", "url": "https://item.jd.com/10073213192312.html"}
    ],
    "笔记本": [
        {"name": "联想小新Pro 16", "url": "https://item.jd.com/100033932037.html"},
        {"name": "Apple MacBook Air", "url": "https://item.jd.com/100071340156.html"},
        {"name": "华为MateBook X Pro", "url": "https://item.jd.com/100049652413.html"},
        {"name": "戴尔灵越16 Plus", "url": "https://item.jd.com/100052772140.html"},
        {"name": "惠普星 15", "url": "https://item.jd.com/100046631794.html"}
    ],
    "平板": [
        {"name": "iPad Air", "url": "https://item.jd.com/10072224836129.html"},
        {"name": "华为MatePad Pro", "url": "https://item.jd.com/100026571596.html"},
        {"name": "小米平板6 Pro", "url": "https://item.jd.com/10078140024885.html"},
        {"name": "三星Galaxy Tab S9", "url": "https://item.jd.com/10080616701216.html"},
        {"name": "vivo Pad 3 Pro", "url": "https://item.jd.com/10087693218339.html"}
    ],
    "显卡": [
        {"name": "NVIDIA RTX 4090", "url": "https://item.jd.com/100064656984.html"},
        {"name": "AMD RX 7900 XTX", "url": "https://item.jd.com/100055093493.html"},
        {"name": "NVIDIA RTX 4070 Ti", "url": "https://item.jd.com/100043632622.html"},
        {"name": "AMD RX 7800 XT", "url": "https://item.jd.com/100068656870.html"},
        {"name": "NVIDIA RTX 4060 Ti", "url": "https://item.jd.com/10078505921804.html"}
    ],
    "耳机": [
        {"name": "Apple AirPods Pro", "url": "https://item.jd.com/100063093853.html"},
        {"name": "索尼WH-1000XM5", "url": "https://item.jd.com/100035143137.html"},
        {"name": "华为FreeBuds Pro 3", "url": "https://item.jd.com/100055276454.html"},
        {"name": "小米Redmi Buds 4 Pro", "url": "https://item.jd.com/100035232997.html"},
        {"name": "森海塞尔Momentum 4", "url": "https://item.jd.com/100049493109.html"}
    ]
}

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='京东多品类商品爬虫')
    
    parser.add_argument('-c', '--category', type=str,
                        choices=list(JD_CATEGORIES.keys()),
                        help=f'要爬取的商品类别，可选值: {", ".join(JD_CATEGORIES.keys())}')
    
    parser.add_argument('-a', '--all', action='store_true',
                        default=False,
                        help='爬取所有类别的商品')
    
    parser.add_argument('-o', '--output', type=str,
                        default=os.path.join("results", "categories"),
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
    
    parser.add_argument('--custom-url', type=str,
                        default=None,
                        help='添加自定义商品URL进行爬取')
    
    parser.add_argument('--max-products', type=int,
                        default=None,
                        help='每个类别最多爬取的商品数量')

    parser.add_argument('--delay-min', type=float,
                        default=SPIDER_CONFIG['BEHAVIOR']['MIN_DELAY'],
                        help='最小爬取间隔（秒）')
    
    parser.add_argument('--delay-max', type=float,
                        default=SPIDER_CONFIG['BEHAVIOR']['MAX_DELAY'],
                        help='最大爬取间隔（秒）')
    
    parser.add_argument('--summary', action='store_true',
                        default=False,
                        help='只生成数据摘要，不保存完整数据')
    
    return parser.parse_args()

def create_output_dir(base_dir, category):
    """创建输出目录"""
    # 创建类别子目录
    category_dir = os.path.join(base_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir, exist_ok=True)
    
    # 创建带有时间戳的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(category_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def crawl_category(category_name, products, args):
    """爬取指定类别的商品"""
    logger.info("=" * 70)
    logger.info(f"开始爬取类别: {Colors.CYAN}{category_name}{Colors.RESET}")
    logger.info(f"商品数量: {len(products)}")
    logger.info("-" * 70)
    
    # 创建输出目录
    output_dir = create_output_dir(args.output, category_name)
    logger.info(f"输出目录: {output_dir}")
    
    # 限制商品数量
    if args.max_products and args.max_products > 0:
        products = products[:args.max_products]
        logger.info(f"已限制爬取数量: {args.max_products}")
    
    # 生成概览文件
    overview_data = {
        "category": category_name,
        "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_products": len(products),
        "products": []
    }
    
    # 遍历爬取每个商品
    success_count = 0
    for idx, product in enumerate(products):
        try:
            product_name = product["name"]
            product_url = product["url"]
            
            logger.info("-" * 70)
            logger.info(f"正在爬取 [{idx+1}/{len(products)}]: {Colors.YELLOW}{product_name}{Colors.RESET}")
            logger.info(f"商品URL: {product_url}")
            
            # 更新配置
            class Args:
                pass
            
            config_args = Args()
            config_args.url = product_url
            config_args.output = output_dir
            config_args.headless = args.headless
            config_args.login = args.login
            config_args.username = args.username
            config_args.password = args.password
            
            # 更新爬虫配置
            update_config(config_args)
            
            # 设置行为延迟
            SPIDER_CONFIG['BEHAVIOR']['MIN_DELAY'] = args.delay_min
            SPIDER_CONFIG['BEHAVIOR']['MAX_DELAY'] = args.delay_max
            SPIDER_CONFIG['BEHAVIOR']['RANDOM_DELAY'] = True
            
            # 运行爬虫
            start_time = time.time()
            logger.info(f"爬取开始: {datetime.now().strftime('%H:%M:%S')}")
            
            result = run_spider(product_url)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"爬取完成: {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"耗时: {duration:.2f}秒")
            
            # 更新概览数据
            if result:
                success_count += 1
                product_data = {
                    "name": product_name,
                    "url": product_url,
                    "id": result.get("product_id", ""),
                    "price": result.get("price", {}).get("current", ""),
                    "brand": result.get("brand", ""),
                    "crawl_success": True
                }
            else:
                product_data = {
                    "name": product_name,
                    "url": product_url,
                    "crawl_success": False,
                    "error": "爬取失败"
                }
            
            overview_data["products"].append(product_data)
            
            # 随机等待一段时间，避免被反爬
            if idx < len(products) - 1:
                wait_time = random.uniform(args.delay_min, args.delay_max)
                logger.info(f"等待 {wait_time:.2f} 秒后继续...")
                time.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"爬取商品失败: {product['url']}, 错误: {e}")
            # 更新概览数据
            overview_data["products"].append({
                "name": product["name"],
                "url": product["url"],
                "crawl_success": False,
                "error": str(e)
            })
            continue
    
    # 更新成功率
    overview_data["success_count"] = success_count
    overview_data["success_rate"] = f"{(success_count / len(products) * 100):.2f}%" if products else "0%"
    
    # 保存概览数据
    overview_path = os.path.join(output_dir, f"overview_{category_name}.json")
    with open(overview_path, "w", encoding="utf-8") as f:
        json.dump(overview_data, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 70)
    logger.info(f"{category_name}类别爬取完成")
    logger.info(f"成功: {success_count}/{len(products)}, 成功率: {overview_data['success_rate']}")
    logger.info(f"概览保存至: {overview_path}")
    logger.info("=" * 70)
    
    return success_count, len(products)

def display_categories_table():
    """显示可选类别表格"""
    print("\n京东商品类别列表:")
    print("=" * 60)
    print(f"{'类别':<10} {'商品数量':<10} {'示例商品'}")
    print("-" * 60)
    
    for category, products in JD_CATEGORIES.items():
        print(f"{category:<10} {len(products):<10} {products[0]['name']}")
    
    print("=" * 60)
    print(f"使用 -c 参数指定要爬取的类别，例如: -c 手机")
    print(f"使用 -a 参数爬取所有类别")
    print("=" * 60)

def crawl_categories(args):
    """爬取选定的商品类别"""
    # 显示开始信息
    print(f"\n{Colors.CYAN}京东多品类商品爬虫启动{Colors.RESET}")
    print("=" * 70)
    
    total_success = 0
    total_products = 0
    
    # 如果未指定类别也未选择全部，则显示类别列表并退出
    if not args.category and not args.all and not args.custom_url:
        display_categories_table()
        return
    
    # 创建主输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    # 处理自定义URL
    if args.custom_url:
        custom_category = "custom"
        custom_products = [{"name": "自定义商品", "url": args.custom_url}]
        success, total = crawl_category(custom_category, custom_products, args)
        total_success += success
        total_products += total
    
    # 处理特定类别
    elif args.category:
        if args.category in JD_CATEGORIES:
            products = JD_CATEGORIES[args.category]
            success, total = crawl_category(args.category, products, args)
            total_success += success
            total_products += total
        else:
            logger.error(f"未知类别: {args.category}")
            print(f"{Colors.RED}错误: 未知类别 {args.category}{Colors.RESET}")
            display_categories_table()
    
    # 处理所有类别
    elif args.all:
        for category, products in JD_CATEGORIES.items():
            success, total = crawl_category(category, products, args)
            total_success += success
            total_products += total
    
    # 显示总结信息
    if total_products > 0:
        print("\n" + "=" * 70)
        print(f"{Colors.GREEN}京东多品类爬虫运行完成{Colors.RESET}")
        print(f"总爬取商品: {total_products}")
        print(f"成功爬取: {total_success}")
        print(f"总成功率: {(total_success / total_products * 100):.2f}%")
        print(f"结果保存至: {os.path.abspath(args.output)}")
        print("=" * 70)

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置日志级别
        logging.basicConfig(level=logging.INFO)
        
        # 爬取商品
        crawl_categories(args)
        
        return 0
    except KeyboardInterrupt:
        logger.info("用户中断爬虫")
        print(f"\n{Colors.YELLOW}用户中断，程序退出{Colors.RESET}")
        return 1
    except Exception as e:
        logger.error(f"爬虫运行出错: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n{Colors.RED}程序错误: {e}{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())