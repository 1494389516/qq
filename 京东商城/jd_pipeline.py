import os
import json
import csv
import logging
from datetime import datetime
from scrapy.exporters import CsvItemExporter
from scrapy import signals
from scrapy.pipelines.images import ImagesPipeline
import scrapy
from config import SPIDER_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JDImagesPipeline(ImagesPipeline):
    """处理京东商品图片的Pipeline"""
    
    def get_media_requests(self, item, info):
        if SPIDER_CONFIG['OUTPUT']['SAVE_IMAGES'] and item.get('images'):
            for image_url in item['images']:
                yield scrapy.Request(image_url)
    
    def file_path(self, request, response=None, info=None, *, item=None):
        # 使用商品ID作为文件夹名，图片URL的MD5作为文件名
        image_guid = request.url.split('/')[-1]
        product_id = item['product_id']
        return f'{product_id}/{image_guid}'
    
    def item_completed(self, results, item, info):
        """处理图片下载完成后的回调"""
        # 收集成功下载的图片路径
        image_paths = [x['path'] for ok, x in results if ok]
        if image_paths:
            item['image_paths'] = image_paths
            logger.info(f"成功下载 {len(image_paths)}/{len(item.get('images', []))} 张图片")
        else:
            item['image_paths'] = []
            if item.get('images'):
                logger.warning(f"未能成功下载任何图片，共 {len(item.get('images', []))} 张")
        return item

class JDProductPipeline:
    """处理京东商品数据的Pipeline"""
    
    def __init__(self):
        self.files = {}
        self.exporters = {}
        self.batch_mode = SPIDER_CONFIG.get('BATCH_MODE', False)
        self.batch_items = []
        self.error_count = 0
        self.success_count = 0
    
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        crawler.signals.connect(pipeline.spider_opened, signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signals.spider_closed)
        return pipeline
    
    def spider_opened(self, spider):
        # 创建保存结果的目录
        results_dir = SPIDER_CONFIG['OUTPUT']['RESULTS_DIR']
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 创建CSV文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = open(f'{results_dir}/jd_products_{timestamp}.csv', 'wb')
        self.files[spider] = csv_file
        
        # 配置CSV导出器
        exporter = CsvItemExporter(csv_file, encoding='utf-8')
        # 扩展导出字段
        exporter.fields_to_export = [
            'product_id', 'title', 'price', 'original_price', 'discount',
            'shop_name', 'shop_score', 'brand', 'model', 
            'category', 'description', 'comments_count', 
            'good_rate', 'stock_status', 'url', 'crawl_time'
        ]
        self.exporters[spider] = exporter
        self.exporters[spider].start_exporting()
        
        # 创建JSON文件（用于保存更详细的数据，如规格和图片）
        self.json_file = open(f'{results_dir}/jd_products_{timestamp}.json', 'w', encoding='utf-8')
        self.json_file.write('[\n')
        self.first_item = True
        
        # 记录开始时间
        self.start_time = datetime.now()
        logger.info(f"Pipeline初始化完成，结果将保存到: {results_dir}")
    
    def spider_closed(self, spider):
        # 关闭CSV导出器
        self.exporters[spider].finish_exporting()
        self.files[spider].close()
        
        # 关闭JSON文件
        self.json_file.write('\n]')
        self.json_file.close()
        
        # 计算运行时间
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # 输出统计信息
        logger.info(f"爬取完成，共处理 {self.success_count + self.error_count} 个商品")
        logger.info(f"成功: {self.success_count}, 失败: {self.error_count}")
        logger.info(f"总耗时: {duration:.2f} 秒")
        
        print(f"\n数据已保存到 {SPIDER_CONFIG['OUTPUT']['RESULTS_DIR']} 目录")
        print(f"成功处理: {self.success_count} 个商品")
        if self.error_count > 0:
            print(f"处理失败: {self.error_count} 个商品")
        print(f"总耗时: {duration:.2f} 秒")
    
    def validate_item(self, item):
        """验证数据项的完整性和有效性"""
        required_fields = ['product_id', 'title', 'url']
        missing_fields = [field for field in required_fields if not item.get(field)]
        
        if missing_fields:
            logger.warning(f"数据缺失关键字段: {', '.join(missing_fields)}")
            return False
        
        # 检查价格字段
        if item.get('price') == '提取失败' or item.get('price') == '需要登录查看':
            logger.warning(f"商品 {item.get('product_id')} 价格提取失败")
            # 不返回False，因为这种情况仍然可以保存其他数据
        
        return True
    
    def clean_item(self, item):
        """清理和规范化数据项"""
        # 确保所有字符串字段不包含首尾空白
        for key, value in item.items():
            if isinstance(value, str):
                item[key] = value.strip()
        
        # 处理价格字段，移除货币符号
        if 'price' in item and isinstance(item['price'], str) and item['price'] not in ['提取失败', '需要登录查看']:
            item['price'] = item['price'].replace('¥', '').replace(',', '')
        
        if 'original_price' in item and isinstance(item['original_price'], str):
            item['original_price'] = item['original_price'].replace('¥', '').replace(',', '')
        
        # 确保评价数和好评率格式一致
        if 'comments_count' in item and isinstance(item['comments_count'], str):
            # 移除"万"等单位，统一为数字
            if '万' in item['comments_count']:
                try:
                    count = float(item['comments_count'].replace('万', '')) * 10000
                    item['comments_count'] = str(int(count))
                except:
                    pass
        
        if 'good_rate' in item and isinstance(item['good_rate'], str):
            # 确保百分比格式一致
            if '%' not in item['good_rate']:
                item['good_rate'] = item['good_rate'] + '%'
        
        return item
    
    def process_item(self, item, spider):
        try:
            # 验证数据
            is_valid = self.validate_item(item)
            if not is_valid:
                logger.warning(f"跳过无效数据项: {item.get('product_id', 'unknown')}")
                self.error_count += 1
                return item
            
            # 清理数据
            item = self.clean_item(item)
            
            # 导出到CSV
            self.exporters[spider].export_item(item)
            
            # 导出到JSON
            if not self.first_item:
                self.json_file.write(',\n')
            else:
                self.first_item = False
            
            # 将item转换为dict并写入JSON
            item_dict = dict(item)
            json_data = json.dumps(item_dict, ensure_ascii=False)
            self.json_file.write(json_data)
            
            # 如果是批量模式，保存到批量列表
            if self.batch_mode:
                self.batch_items.append(item_dict)
            
            # 记录成功处理
            self.success_count += 1
            logger.info(f"成功处理商品: {item.get('product_id')} - {item.get('title', '')[:30]}")
            
            # 单独保存每个商品的详细数据
            self.save_individual_product(item_dict)
            
            return item
        except Exception as e:
            self.error_count += 1
            logger.error(f"处理数据项时出错: {e}")
            return item
    
    def save_individual_product(self, item):
        """为每个商品单独保存一个JSON文件"""
        try:
            # 确保有商品ID
            if not item.get('product_id'):
                return
            
            # 创建保存目录
            results_dir = SPIDER_CONFIG['OUTPUT']['RESULTS_DIR']
            product_dir = os.path.join(results_dir, item['product_id'])
            if not os.path.exists(product_dir):
                os.makedirs(product_dir)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"product_{item['product_id']}_{timestamp}"
            
            # 保存JSON
            with open(f'{product_dir}/{filename}.json', 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
            
            # 保存CSV
            with open(f'{product_dir}/{filename}.csv', 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(item.keys())
                # 写入数据
                writer.writerow(item.values())
            
            logger.info(f"已保存商品 {item['product_id']} 的详细数据")
        except Exception as e:
            logger.error(f"保存单个商品数据时出错: {e}")
    
    def get_batch_results(self):
        """获取批量处理的结果"""
        return {
            'items': self.batch_items,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'total_count': self.success_count + self.error_count
        }

# 扩展的Pipeline，用于处理商品评论
class JDCommentsPipeline:
    """处理京东商品评论的Pipeline"""
    
    def __init__(self):
        self.comments = {}
    
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        crawler.signals.connect(pipeline.spider_opened, signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signals.spider_closed)
        return pipeline
    
    def spider_opened(self, spider):
        logger.info("评论Pipeline初始化完成")
    
    def spider_closed(self, spider):
        # 保存所有评论
        for product_id, comments in self.comments.items():
            self.save_comments(product_id, comments)
        
        logger.info(f"评论Pipeline关闭，共处理 {len(self.comments)} 个商品的评论")
    
    def process_item(self, item, spider):
        # 如果item包含评论数据，则处理
        if 'comments' in item and item['comments']:
            product_id = item.get('product_id', 'unknown')
            self.comments[product_id] = item['comments']
            logger.info(f"收集到商品 {product_id} 的 {len(item['comments'])} 条评论")
        
        return item
    
    def save_comments(self, product_id, comments):
        """保存商品评论"""
        try:
            # 创建保存目录
            results_dir = SPIDER_CONFIG['OUTPUT']['RESULTS_DIR']
            comments_dir = os.path.join(results_dir, product_id, 'comments')
            if not os.path.exists(comments_dir):
                os.makedirs(comments_dir)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comments_{product_id}_{timestamp}"
            
            # 保存JSON
            with open(f'{comments_dir}/{filename}.json', 'w', encoding='utf-8') as f:
                json.dump(comments, f, ensure_ascii=False, indent=2)
            
            # 保存CSV
            with open(f'{comments_dir}/{filename}.csv', 'w', encoding='utf-8', newline='') as f:
                if comments and isinstance(comments[0], dict):
                    fieldnames = comments[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(comments)
            
            logger.info(f"已保存商品 {product_id} 的 {len(comments)} 条评论")
        except Exception as e:
            logger.error(f"保存评论数据时出错: {e}")