#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东商品数据可视化工具
用于分析和可视化爬取的京东商品数据
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='京东商品数据可视化工具')
    
    parser.add_argument('-d', '--data-dir', type=str,
                        default="results",
                        help='数据目录，包含爬取的商品数据')
    
    parser.add_argument('-c', '--category', type=str,
                        default=None,
                        help='要分析的商品类别')
    
    parser.add_argument('-o', '--output', type=str,
                        default="analysis",
                        help='分析结果输出目录')
    
    parser.add_argument('--price-range', action='store_true',
                        default=False,
                        help='分析价格区间分布')
    
    parser.add_argument('--brand-compare', action='store_true',
                        default=False,
                        help='比较不同品牌的商品')
    
    parser.add_argument('--save-format', type=str,
                        choices=['png', 'jpg', 'pdf', 'svg'],
                        default='png',
                        help='图表保存格式')
    
    parser.add_argument('--interactive', action='store_true',
                        default=False,
                        help='启用交互模式')
    
    return parser.parse_args()

def load_product_data(data_dir, category=None):
    """加载商品数据
    
    Args:
        data_dir: 数据目录
        category: 类别名称
    
    Returns:
        pandas.DataFrame: 商品数据
    """
    data_dir = Path(data_dir)
    all_data = []
    
    # 如果指定了类别，只搜索该类别目录
    if category:
        category_dir = data_dir / category
        if not category_dir.exists():
            logger.error(f"类别目录不存在: {category_dir}")
            return pd.DataFrame()
        search_dirs = [category_dir]
    else:
        # 搜索所有可能的类别目录
        search_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        # 如果有categories子目录，优先处理它
        categories_dir = data_dir / "categories"
        if categories_dir.exists() and categories_dir.is_dir():
            category_subdirs = [d for d in categories_dir.iterdir() if d.is_dir()]
            search_dirs.extend(category_subdirs)
    
    # 遍历所有目录查找JSON文件
    for directory in search_dirs:
        # 获取该目录的类别名称
        category_name = directory.name
        
        # 递归搜索所有JSON文件
        for json_file in directory.glob('**/*.json'):
            # 跳过overview文件
            if 'overview' in json_file.name:
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    product_data = json.load(f)
                
                # 添加类别信息
                if isinstance(product_data, dict):
                    # 如果没有类别信息，添加目录名作为类别
                    if 'category' not in product_data:
                        product_data['category'] = category_name
                    
                    # 添加到数据列表
                    all_data.append(product_data)
            except Exception as e:
                logger.warning(f"加载文件失败 {json_file}: {e}")
    
    # 如果没有数据，返回空DataFrame
    if not all_data:
        logger.warning("未找到任何商品数据")
        return pd.DataFrame()
    
    # 转换为DataFrame
    try:
        # 提取关键字段到扁平结构
        flat_data = []
        for product in all_data:
            item = {
                'product_id': product.get('product_id', ''),
                'title': product.get('title', ''),
                'brand': product.get('brand', ''),
                'category': product.get('category', ''),
                'crawl_time': product.get('crawl_time', ''),
            }
            
            # 提取价格
            price_info = product.get('price', {})
            if isinstance(price_info, dict):
                item['price'] = price_info.get('current', '')
                item['original_price'] = price_info.get('original', '')
            
            # 提取评分信息
            comments = product.get('comments', {})
            if isinstance(comments, dict):
                item['rating'] = comments.get('average_rating', '')
                item['comment_count'] = comments.get('count', '')
            
            flat_data.append(item)
        
        # 创建DataFrame
        df = pd.DataFrame(flat_data)
        
        # 处理价格列，移除非数字字符
        if 'price' in df.columns:
            df['price'] = df['price'].apply(lambda x: 
                float(re.sub(r'[^\d.]', '', str(x))) if pd.notna(x) and str(x).strip() else np.nan)
        
        if 'original_price' in df.columns:
            df['original_price'] = df['original_price'].apply(lambda x: 
                float(re.sub(r'[^\d.]', '', str(x))) if pd.notna(x) and str(x).strip() else np.nan)
        
        # 处理评分和评论数量
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        if 'comment_count' in df.columns:
            df['comment_count'] = df['comment_count'].apply(lambda x: 
                float(re.sub(r'[^\d.]', '', str(x))) if pd.notna(x) and str(x).strip() else np.nan)
        
        return df
    
    except Exception as e:
        logger.error(f"处理数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def analyze_price_distribution(df, output_dir, save_format='png'):
    """分析价格分布
    
    Args:
        df: 商品数据DataFrame
        output_dir: 输出目录
        save_format: 图表保存格式
    """
    if 'price' not in df.columns or df['price'].isnull().all():
        logger.warning("没有有效的价格数据可供分析")
        return
    
    # 去除异常值
    df_clean = df[df['price'] > 0]
    df_clean = df_clean[df_clean['price'] < df_clean['price'].quantile(0.99)]  # 移除最高1%的价格
    
    if df_clean.empty:
        logger.warning("清理异常值后没有有效数据")
        return
    
    # 设置图表样式
    sns.set_style("whitegrid")
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 价格分布直方图
    plt.subplot(2, 2, 1)
    sns.histplot(df_clean['price'], kde=True, bins=30)
    plt.title('商品价格分布')
    plt.xlabel('价格 (元)')
    plt.ylabel('商品数量')
    
    # 按类别的价格箱型图
    if 'category' in df_clean.columns and len(df_clean['category'].unique()) > 1:
        plt.subplot(2, 2, 2)
        sns.boxplot(x='category', y='price', data=df_clean)
        plt.title('不同类别商品价格分布')
        plt.xlabel('商品类别')
        plt.ylabel('价格 (元)')
        plt.xticks(rotation=45)
    
    # 按品牌的价格箱型图
    if 'brand' in df_clean.columns and len(df_clean['brand'].unique()) > 1:
        # 获取前10个品牌
        top_brands = df_clean['brand'].value_counts().nlargest(10).index
        brand_data = df_clean[df_clean['brand'].isin(top_brands)]
        
        plt.subplot(2, 2, 3)
        sns.boxplot(x='brand', y='price', data=brand_data)
        plt.title('主要品牌价格分布')
        plt.xlabel('品牌')
        plt.ylabel('价格 (元)')
        plt.xticks(rotation=45)
    
    # 价格分布密度图
    plt.subplot(2, 2, 4)
    sns.kdeplot(df_clean['price'], fill=True)
    plt.title('价格密度曲线')
    plt.xlabel('价格 (元)')
    plt.ylabel('密度')
    
    # 布局调整
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, f'price_distribution.{save_format}')
    plt.savefig(output_path, dpi=300)
    logger.info(f"价格分布分析已保存至: {output_path}")
    
    # 输出价格统计信息
    print("\n价格统计信息:")
    print("=" * 50)
    price_stats = df_clean['price'].describe()
    print(f"商品数量: {len(df_clean)}")
    print(f"平均价格: {price_stats['mean']:.2f}元")
    print(f"中位价格: {price_stats['50%']:.2f}元")
    print(f"最低价格: {price_stats['min']:.2f}元")
    print(f"最高价格: {price_stats['max']:.2f}元")
    print(f"价格标准差: {price_stats['std']:.2f}元")
    print("=" * 50)
    
    # 按类别汇总价格
    if 'category' in df_clean.columns and len(df_clean['category'].unique()) > 1:
        print("\n各类别价格统计:")
        print("=" * 50)
        category_stats = df_clean.groupby('category')['price'].agg(['count', 'mean', 'median', 'min', 'max'])
        print(category_stats.to_string())
        print("=" * 50)

def analyze_brand_comparison(df, output_dir, save_format='png'):
    """比较不同品牌的商品数据
    
    Args:
        df: 商品数据DataFrame
        output_dir: 输出目录
        save_format: 图表保存格式
    """
    if 'brand' not in df.columns or df['brand'].isnull().all():
        logger.warning("没有有效的品牌数据可供分析")
        return
    
    # 获取前10个最常见品牌
    brand_counts = df['brand'].value_counts()
    top_brands = brand_counts.nlargest(10).index
    df_top_brands = df[df['brand'].isin(top_brands)]
    
    if df_top_brands.empty:
        logger.warning("筛选品牌后没有有效数据")
        return
    
    # 设置图表样式
    sns.set_style("whitegrid")
    
    # 创建图表
    plt.figure(figsize=(12, 12))
    
    # 品牌数量统计
    plt.subplot(2, 2, 1)
    sns.barplot(x=brand_counts.nlargest(10).index, y=brand_counts.nlargest(10).values)
    plt.title('商品数量最多的品牌')
    plt.xlabel('品牌')
    plt.ylabel('商品数量')
    plt.xticks(rotation=45)
    
    # 品牌价格对比
    if 'price' in df.columns and not df['price'].isnull().all():
        plt.subplot(2, 2, 2)
        sns.barplot(x='brand', y='price', data=df_top_brands, estimator=np.mean)
        plt.title('各品牌平均价格')
        plt.xlabel('品牌')
        plt.ylabel('平均价格 (元)')
        plt.xticks(rotation=45)
    
    # 评分对比
    if 'rating' in df.columns and not df['rating'].isnull().all():
        plt.subplot(2, 2, 3)
        sns.barplot(x='brand', y='rating', data=df_top_brands)
        plt.title('各品牌平均评分')
        plt.xlabel('品牌')
        plt.ylabel('平均评分')
        plt.xticks(rotation=45)
    
    # 评论数对比
    if 'comment_count' in df.columns and not df['comment_count'].isnull().all():
        plt.subplot(2, 2, 4)
        sns.barplot(x='brand', y='comment_count', data=df_top_brands)
        plt.title('各品牌平均评论数')
        plt.xlabel('品牌')
        plt.ylabel('平均评论数')
        plt.xticks(rotation=45)
    
    # 布局调整
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, f'brand_comparison.{save_format}')
    plt.savefig(output_path, dpi=300)
    logger.info(f"品牌对比分析已保存至: {output_path}")
    
    # 输出品牌统计信息
    print("\n品牌统计信息:")
    print("=" * 70)
    print(f"{'品牌':<15} {'商品数量':<10} {'平均价格':<15} {'最低价格':<15} {'最高价格':<15}")
    print("-" * 70)
    
    for brand in top_brands:
        brand_data = df[df['brand'] == brand]
        count = len(brand_data)
        
        if 'price' in brand_data.columns and not brand_data['price'].isnull().all():
            avg_price = brand_data['price'].mean()
            min_price = brand_data['price'].min()
            max_price = brand_data['price'].max()
            print(f"{brand:<15} {count:<10} {avg_price:<15.2f} {min_price:<15.2f} {max_price:<15.2f}")
        else:
            print(f"{brand:<15} {count:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("=" * 70)

def analyze_data(df, output_dir, args):
    """分析数据并生成可视化
    
    Args:
        df: 商品数据DataFrame
        output_dir: 输出目录
        args: 命令行参数
    """
    if df.empty:
        logger.error("没有数据可供分析")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 基本信息
    print("\n" + "="*70)
    print(f"{Colors.CYAN}京东商品数据分析报告{Colors.RESET}")
    print("=" * 70)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据记录数: {len(df)}")
    
    if 'category' in df.columns:
        print(f"商品类别数: {len(df['category'].unique())}")
        print(f"类别分布: {dict(df['category'].value_counts())}")
    
    if 'brand' in df.columns:
        print(f"品牌数量: {len(df['brand'].unique())}")
    
    if 'price' in df.columns and not df['price'].isnull().all():
        print(f"价格范围: {df['price'].min():.2f} - {df['price'].max():.2f}元")
    
    print("=" * 70)
    
    # 根据参数执行特定分析
    if args.price_range:
        analyze_price_distribution(df, output_dir, args.save_format)
    
    if args.brand_compare:
        analyze_brand_comparison(df, output_dir, args.save_format)
    
    # 如果没有指定具体分析，执行所有分析
    if not args.price_range and not args.brand_compare:
        analyze_price_distribution(df, output_dir, args.save_format)
        analyze_brand_comparison(df, output_dir, args.save_format)
    
    # 保存数据摘要
    summary_path = os.path.join(output_dir, 'data_summary.csv')
    df.to_csv(summary_path, index=False)
    logger.info(f"数据摘要已保存至: {summary_path}")

def show_interactive_mode(df, output_dir, save_format):
    """交互式数据分析模式
    
    Args:
        df: 商品数据DataFrame
        output_dir: 输出目录
        save_format: 图表保存格式
    """
    while True:
        print("\n" + "="*50)
        print(f"{Colors.CYAN}京东商品数据交互式分析{Colors.RESET}")
        print("=" * 50)
        print("1. 显示数据基本信息")
        print("2. 价格分布分析")
        print("3. 品牌对比分析")
        print("4. 自定义数据筛选")
        print("5. 保存当前数据")
        print("0. 退出")
        print("=" * 50)
        
        choice = input("请选择操作 [0-5]: ")
        
        if choice == '0':
            break
        
        elif choice == '1':
            print("\n数据基本信息:")
            print("=" * 50)
            print(f"记录数: {len(df)}")
            print(f"列名: {', '.join(df.columns)}")
            print("\n数据统计:")
            print(df.describe().to_string())
            print("\n前5条记录:")
            print(df.head().to_string())
        
        elif choice == '2':
            analyze_price_distribution(df, output_dir, save_format)
            
            # 显示图表
            plt.show()
        
        elif choice == '3':
            analyze_brand_comparison(df, output_dir, save_format)
            
            # 显示图表
            plt.show()
        
        elif choice == '4':
            # 自定义筛选
            print("\n可用于筛选的列:")
            for i, col in enumerate(df.columns):
                print(f"{i+1}. {col}")
            
            col_idx = input("\n选择要筛选的列 [1-{}] (输入q返回): ".format(len(df.columns)))
            
            if col_idx.lower() == 'q':
                continue
            
            try:
                col_idx = int(col_idx)
                if col_idx < 1 or col_idx > len(df.columns):
                    print(f"{Colors.RED}无效的选择!{Colors.RESET}")
                    continue
                
                column = df.columns[col_idx-1]
                print(f"\n'{column}'列的值分布:")
                
                if df[column].dtype == 'object':
                    # 字符串列显示前10个最常见的值
                    print(df[column].value_counts().nlargest(10))
                else:
                    # 数值列显示统计信息
                    print(df[column].describe())
                
                # 筛选方式
                print("\n筛选方式:")
                print("1. 等于特定值")
                print("2. 大于特定值")
                print("3. 小于特定值")
                print("4. 包含特定文本")
                
                filter_type = input("\n选择筛选方式 [1-4]: ")
                
                value = input("输入筛选值: ")
                
                if filter_type == '1':
                    filtered_df = df[df[column] == value]
                elif filter_type == '2':
                    filtered_df = df[df[column] > float(value)]
                elif filter_type == '3':
                    filtered_df = df[df[column] < float(value)]
                elif filter_type == '4':
                    filtered_df = df[df[column].astype(str).str.contains(value)]
                else:
                    print(f"{Colors.RED}无效的选择!{Colors.RESET}")
                    continue
                
                print(f"\n筛选结果: {len(filtered_df)}条记录")
                
                if not filtered_df.empty:
                    print("\n前5条记录:")
                    print(filtered_df.head().to_string())
                    
                    # 询问是否使用筛选后的数据
                    use_filtered = input("\n是否使用筛选后的数据进行分析? (y/n): ")
                    if use_filtered.lower() == 'y':
                        df = filtered_df
                        print(f"{Colors.GREEN}已更新数据集，现在使用{len(df)}条记录.{Colors.RESET}")
                
            except Exception as e:
                print(f"{Colors.RED}筛选出错: {e}{Colors.RESET}")
        
        elif choice == '5':
            # 保存数据
            filename = input("输入保存的文件名 (不含扩展名): ")
            if not filename:
                filename = 'custom_data'
            
            save_path = os.path.join(output_dir, f"{filename}.csv")
            df.to_csv(save_path, index=False)
            print(f"{Colors.GREEN}数据已保存至: {save_path}{Colors.RESET}")
            
            # 保存Excel版本
            try:
                excel_path = os.path.join(output_dir, f"{filename}.xlsx")
                df.to_excel(excel_path, index=False)
                print(f"{Colors.GREEN}Excel数据已保存至: {excel_path}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.YELLOW}Excel保存失败: {e}{Colors.RESET}")
        
        else:
            print(f"{Colors.RED}无效的选择!{Colors.RESET}")
        
        input("\n按回车键继续...")

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置日志级别
        logging.basicConfig(level=logging.INFO)
        
        print(f"\n{Colors.CYAN}京东商品数据可视化分析工具{Colors.RESET}")
        print("=" * 70)
        print(f"数据目录: {args.data_dir}")
        
        # 检查依赖库
        try:
            import matplotlib
            import seaborn
            import pandas
            print(f"{Colors.GREEN}✓ 数据分析环境检查通过{Colors.RESET}")
        except ImportError as e:
            print(f"{Colors.RED}✗ 缺少必要的库: {e}{Colors.RESET}")
            print("请安装必要的依赖: pip install matplotlib seaborn pandas")
            return 1
        
        # 加载数据
        print(f"正在加载数据，请稍候...")
        df = load_product_data(args.data_dir, args.category)
        
        if df.empty:
            print(f"{Colors.RED}错误: 未找到有效的商品数据{Colors.RESET}")
            return 1
        
        print(f"{Colors.GREEN}✓ 已加载{len(df)}条商品数据{Colors.RESET}")
        
        # 创建输出目录
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
        
        # 交互模式
        if args.interactive:
            show_interactive_mode(df, args.output, args.save_format)
        else:
            # 分析数据
            analyze_data(df, args.output, args)
        
        print(f"\n{Colors.GREEN}分析完成! 结果保存在: {os.path.abspath(args.output)}{Colors.RESET}")
        return 0
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}用户中断，程序退出{Colors.RESET}")
        return 1
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n{Colors.RED}程序错误: {e}{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 