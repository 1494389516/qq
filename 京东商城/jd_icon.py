#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成京东爬虫的图标文件
"""

import os
import base64
import tempfile
from PIL import Image, ImageDraw, ImageFont

# 京东红色
JD_RED = (227, 28, 36)
# 背景色
BG_COLOR = (255, 255, 255)

def create_jd_icon(output_path='jd_icon.ico', size=256):
    """创建京东爬虫图标
    
    Args:
        output_path: 输出图标路径
        size: 图标大小
    """
    print(f"正在生成图标: {output_path}")
    
    # 创建一个正方形图像
    img = Image.new('RGB', (size, size), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # 计算边距
    margin = size // 10
    
    # 绘制圆角矩形背景
    radius = size // 8
    draw.rounded_rectangle(
        [(margin, margin), (size - margin, size - margin)], 
        radius=radius, 
        fill=JD_RED
    )
    
    # 尝试加载字体
    font_size = size // 2
    try:
        # 尝试使用系统字体
        font_path = None
        
        # 检查常见字体路径
        font_candidates = [
            # Windows字体
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simfang.ttf", # 仿宋
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            # macOS字体
            "/System/Library/Fonts/PingFang.ttc",
            # Linux字体
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        
        for candidate in font_candidates:
            if os.path.exists(candidate):
                font_path = candidate
                break
        
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 如果找不到合适的字体，使用默认字体
            font = ImageFont.load_default()
            font_size = size // 4  # 默认字体较小，调整大小
    except Exception as e:
        print(f"加载字体出错: {e}")
        font = ImageFont.load_default()
        font_size = size // 4
    
    # 在中心绘制"JD"文字
    text = "JD"
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (font_size * 2, font_size)
    position = ((size - text_width) // 2, (size - text_height) // 2)
    draw.text(position, text, font=font, fill=BG_COLOR)
    
    # 绘制爬虫图标（简单的蜘蛛形状）
    spider_size = size // 4
    spider_pos = (size - spider_size - margin // 2, margin // 2)
    
    # 蜘蛛身体
    draw.ellipse(
        [(spider_pos[0], spider_pos[1]), 
         (spider_pos[0] + spider_size, spider_pos[1] + spider_size)], 
        fill=BG_COLOR
    )
    
    # 蜘蛛腿
    leg_length = spider_size // 2
    center_x = spider_pos[0] + spider_size // 2
    center_y = spider_pos[1] + spider_size // 2
    
    # 绘制8条腿
    for i in range(8):
        angle = i * 45
        import math
        end_x = center_x + int(leg_length * math.cos(math.radians(angle)))
        end_y = center_y + int(leg_length * math.sin(math.radians(angle)))
        draw.line([(center_x, center_y), (end_x, end_y)], fill=BG_COLOR, width=2)
    
    # 保存为ICO文件
    try:
        # 创建临时PNG文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            img.save(tmp_path)
        
        # 转换为ICO
        img = Image.open(tmp_path)
        img.save(output_path, format='ICO', sizes=[(size, size)])
        
        # 删除临时文件
        os.unlink(tmp_path)
        
        print(f"图标已生成: {output_path}")
        return True
    except Exception as e:
        print(f"保存图标时出错: {e}")
        return False

if __name__ == "__main__":
    create_jd_icon()
    print("完成!") 