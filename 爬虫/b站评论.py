import requests
import json
import time
import random
import urllib.parse
from datetime import datetime
import pandas as pd
import os
import traceback
import pandas as pd

def parse_api_url(api_url):
    """
    解析API URL，提取域名和参数
    :param api_url: API URL或路径
    :return: 完整URL和参数字典
    """
    # 如果是相对路径，添加域名
    if api_url.startswith('/'):
        full_url = f"https://api.bilibili.com{api_url}"
    elif api_url.startswith('http'):
        full_url = api_url
    else:
        full_url = f"https://api.bilibili.com/{api_url}"
    
    # 解析URL，提取参数
    parsed_url = urllib.parse.urlparse(full_url)
    params = dict(urllib.parse.parse_qsl(parsed_url.query))
    
    # 重建不带参数的URL
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    
    return base_url, params

def get_all_comments(api_url, cookies_str, max_pages=100, get_sub_comments=True):
    """
    获取所有评论
    :param api_url: API URL或路径
    :param cookies_str: Cookie字符串
    :param max_pages: 最大获取页数，防止无限循环
    :param get_sub_comments: 是否获取子评论
    :return: 评论列表
    """
    # 解析API URL
    base_url, params = parse_api_url(api_url)
    
    # 打印解析后的URL和参数，帮助调试
    print(f"基础URL: {base_url}")
    print(f"参数: {json.dumps(params, ensure_ascii=False)}")
    
    # 解析Cookie字符串为字典
    cookies = {}
    if cookies_str:
        for item in cookies_str.split(';'):
            if '=' in item:
                key, value = item.strip().split('=', 1)
                cookies[key] = value
        print(f"已设置Cookie，包含 {len(cookies)} 个键值对")
    else:
        print("警告：未提供Cookie，可能无法获取需要登录才能查看的评论")
    
    # 请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.bilibili.com",
        "Referer": "https://www.bilibili.com/",
        "Connection": "keep-alive",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }
    
    # 确保params中包含必要的参数
    if 'type' not in params:
        params['type'] = '1'  # 默认type=1表示视频评论
    if 'sort' not in params:
        params['sort'] = '2'  # 默认按热度排序，1是按时间
    if 'ps' not in params:
        params['ps'] = '30'  # 每页评论数，最大值通常是30
    
    # 存储所有评论
    all_comments = []
    page_count = 1
    
    # 添加重试次数和最大页数限制
    max_retries = 5  # 增加重试次数
    
    try:
        while page_count <= max_pages:
            print(f"\n正在获取第 {page_count} 页评论...")
            
            # 发送请求，添加重试机制
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # 打印当前请求参数
                    if retry_count > 0:
                        print(f"第 {retry_count} 次重试...")
                        # 重试时增加延迟
                        retry_delay = random.uniform(3 + retry_count * 2, 6 + retry_count * 3)
                        print(f"等待 {retry_delay:.1f} 秒后重试...")
                        time.sleep(retry_delay)
                    
                    # 发送请求前的随机延迟
                    if page_count > 1:  # 第一页不延迟
                        pre_request_delay = random.uniform(1, 2)  # 减少基础延迟时间
                        print(f"请求前等待 {pre_request_delay:.1f} 秒...")
                        time.sleep(pre_request_delay)
                        
                    # 如果是重试，增加延迟时间
                    if retry_count > 0:
                        retry_delay = min(2 * retry_count, 10)  # 指数增长但最大10秒
                        print(f"第 {retry_count} 次重试，等待 {retry_delay:.1f} 秒...")
                        time.sleep(retry_delay)
                    
                    # 发送请求
                    response = requests.get(
                        base_url,
                        headers=headers,
                        params=params,
                        cookies=cookies,
                        timeout=15  # 增加超时时间
                    )
                    
                    # 检查HTTP状态码
                    if response.status_code != 200:
                        print(f"HTTP错误: {response.status_code}")
                        if response.status_code in [429, 403]:  # 请求过多或被禁止
                            print("请求被限制，等待更长时间...")
                            time.sleep(random.uniform(10, 20))
                        retry_count += 1
                        continue
                    
                    # 解析JSON
                    try:
                        data = response.json()
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {str(e)}")
                        print(f"响应内容: {response.text[:200]}...")  # 只打印前200个字符
                        retry_count += 1
                        continue
                    
                    # 检查API返回码
                    if data['code'] != 0:
                        print(f"API错误: {data.get('code')} - {data.get('message')}")
                        
                        # 特殊处理各种错误码
                        if data['code'] == -412:  # 请求被拦截
                            print("请求被拦截，可能是请求频率过高，调整策略...")
                            time.sleep(random.uniform(5, 10))  # 适度增加等待时间
                            # 更新请求头的部分字段
                            headers['User-Agent'] = random.choice([
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                            ])
                        elif data['code'] == -403:  # 访问权限不足
                            if not cookies_str:
                                print("访问权限不足，需要登录Cookie...")
                                return all_comments
                            else:
                                print("访问被拒绝，等待更长时间...")
                                time.sleep(random.uniform(10, 20))
                        elif data['code'] == -404:  # 资源不存在
                            print("资源不存在，结束请求")
                            return all_comments
                        elif data['code'] == -401:  # 未登录
                            print("需要登录才能访问，请提供登录Cookie")
                            return all_comments
                        elif data['code'] == -509:  # 请求过于频繁
                            wait_time = random.uniform(20, 40)
                            print(f"请求过于频繁，将等待 {wait_time:.1f} 秒后重试...")
                            time.sleep(wait_time)
                        else:  # 其他错误
                            print(f"未知错误，等待后重试...")
                            time.sleep(random.uniform(3, 6))
                        
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"达到最大重试次数 ({max_retries})，跳过当前页")
                            break
                        continue
                    
                    success = True
                    
                    # 成功获取数据后的短暂延迟
                    post_success_delay = random.uniform(1, 2)
                    print(f"成功获取数据，等待 {post_success_delay:.1f} 秒后继续...")
                    time.sleep(post_success_delay)
                    
                except requests.exceptions.RequestException as e:
                    print(f"请求异常: {str(e)}")
                    retry_count += 1
                    time.sleep(2)
                except json.JSONDecodeError:
                    print("JSON解析错误")
                    retry_count += 1
                    time.sleep(2)
            
            # 如果所有重试都失败，则退出循环
            if not success:
                print("多次尝试后仍然失败，停止获取评论")
                break
            
            # 显示评论总数信息
            if 'data' in data and 'cursor' in data['data']:
                total = data['data']['cursor'].get('all_count', 0)
                print(f"\n评论总数: {total}")
                print(f"当前已获取: {len(all_comments)}")

            # 提取评论信息
            replies = data['data'].get('replies', [])
            if not replies:
                print("没有更多评论了")
                break
            
            for reply in replies:
                # 主评论
                comment = {
                    "用户名": reply['member']['uname'],
                    "内容": reply['content']['message'],
                    "点赞数": reply['like'],
                    "发布时间": datetime.fromtimestamp(reply['ctime']).strftime('%Y-%m-%d %H:%M:%S'),
                    "回复数": reply['rcount'],
                    "用户ID": reply['member']['mid'],
                    "IP属地": reply.get('reply_control', {}).get('location', '未知'),
                    "rpid": reply['rpid'],  # 评论ID，用于获取子评论
                    "子评论": []
                }
                
                # 获取子评论（如果启用）
                if get_sub_comments and reply['rcount'] > 0:
                    try:
                        print(f"  - 获取评论 #{reply['rpid']} 的 {reply['rcount']} 条回复...")
                        
                        # 子评论分页获取
                        sub_page = 1
                        max_sub_pages = (reply['rcount'] + 19) // 20  # 每页20条评论
                        
                        while sub_page <= max_sub_pages:
                            # 构造子评论请求参数
                            reply_params = params.copy()
                            reply_params['root'] = str(reply['rpid'])
                            reply_params['type'] = '1'
                            reply_params['ps'] = '20'  # 每页数量
                            reply_params['pn'] = str(sub_page)  # 页码
                            
                            # 更新时间戳
                            reply_params['wts'] = str(int(time.time()))
                            
                            # 请求子评论
                            retry_count = 0
                            while retry_count < max_retries:
                                try:
                                    reply_response = requests.get(
                                        base_url, 
                                        headers=headers, 
                                        params=reply_params, 
                                        cookies=cookies,
                                        timeout=10
                                    )
                                    reply_data = reply_response.json()
                                    
                                    if reply_data['code'] == 0 and 'replies' in reply_data.get('data', {}):
                                        sub_replies = reply_data['data']['replies'] or []
                                        print(f"  - 成功获取第 {sub_page}/{max_sub_pages} 页回复，本页 {len(sub_replies)} 条")
                                        
                                        for sub_reply in sub_replies:
                                            sub_comment = {
                                                "用户名": sub_reply['member']['uname'],
                                                "内容": sub_reply['content']['message'],
                                                "点赞数": sub_reply['like'],
                                                "发布时间": datetime.fromtimestamp(sub_reply['ctime']).strftime('%Y-%m-%d %H:%M:%S'),
                                                "用户ID": sub_reply['member']['mid'],
                                                "IP属地": sub_reply.get('reply_control', {}).get('location', '未知')
                                            }
                                            comment["子评论"].append(sub_comment)
                                        
                                        break  # 成功获取数据，跳出重试循环
                                    else:
                                        error_code = reply_data.get('code', 'unknown')
                                        error_msg = reply_data.get('message', 'unknown error')
                                        print(f"  - 获取回复失败: {error_code} - {error_msg}")
                                        
                                        if error_code == -412:  # 请求被拒绝
                                            print("  - 请求被拒绝，等待更长时间...")
                                            time.sleep(random.uniform(5, 10))
                                        
                                        retry_count += 1
                                        if retry_count < max_retries:
                                            print(f"  - 第 {retry_count} 次重试...")
                                            time.sleep(random.uniform(2, 4))
                                        
                                except Exception as e:
                                    print(f"  - 请求异常: {str(e)}")
                                    retry_count += 1
                                    if retry_count < max_retries:
                                        print(f"  - 第 {retry_count} 次重试...")
                                        time.sleep(random.uniform(2, 4))
                            
                            if retry_count >= max_retries:
                                print("  - 多次重试后仍然失败，跳过此页回复")
                                break
                            
                            sub_page += 1
                            
                            # 随机延迟，避免请求过快
                            delay = random.uniform(0.8, 1.5)  # 减少子评论请求的延迟时间
                            print(f"  - 等待 {delay:.1f} 秒...")
                            time.sleep(delay)
                        
                    except Exception as e:
                        print(f"  - 获取子评论时出错: {str(e)}")
                
                all_comments.append(comment)
            
            # 获取下一页的偏移量
            if 'cursor' in data['data']:
                # 打印完整的cursor数据，帮助调试
                print(f"调试信息 - cursor数据: {json.dumps(data['data']['cursor'], ensure_ascii=False)}")
                
                # 检查不同可能的分页字段
                pagination_offset = None
                next_page_exists = False
                
                # 尝试从pagination_reply获取
                if 'pagination_reply' in data['data']['cursor']:
                    pagination_offset = data['data']['cursor']['pagination_reply'].get('offset')
                    next_page_exists = data['data']['cursor']['pagination_reply'].get('is_end', 1) == 0
                
                # 尝试从all_reply获取
                elif 'all_reply' in data['data']['cursor']:
                    pagination_offset = data['data']['cursor']['all_reply'].get('offset')
                    next_page_exists = data['data']['cursor']['all_reply'].get('is_end', 1) == 0
                
                # 尝试直接从cursor获取
                elif 'offset' in data['data']['cursor']:
                    pagination_offset = data['data']['cursor'].get('offset')
                    next_page_exists = data['data']['cursor'].get('is_end', 1) == 0
                
                # 检查是否有下一页
                if pagination_offset is not None:  # 修改：只要有offset值就继续，不再检查is_end
                    print(f"获取到下一页offset: {pagination_offset}")
                    
                    # 更新分页参数
                    pagination_str = {
                        "offset": pagination_offset
                    }
                    params["pagination_str"] = json.dumps(pagination_str)
                    
                    # 重要：每次请求都需要更新wts参数
                    current_time = int(time.time())
                    params["wts"] = str(current_time)
                    
                    # 更新其他可能需要的参数
                    if 'next_offset' in data['data']['cursor']:
                        params['next_offset'] = data['data']['cursor']['next_offset']
                    
                    # 如果有mode参数，也更新它
                    if 'mode' in data['data']['cursor']:
                        params['mode'] = data['data']['cursor']['mode']
                    
                    print("成功更新分页参数，准备获取下一页")
                else:
                    print("已到达最后一页（无offset值）")
                    break
            else:
                print("无法获取下一页参数（无cursor数据）")
                break
            
            page_count += 1
            
            # 添加随机延迟，避免请求过快
            time.sleep(random.uniform(1, 3))
        
        print(f"\n共获取到 {len(all_comments)} 条评论")
        return all_comments
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return all_comments  # 返回已获取的评论

if __name__ == "__main__":
    
    # 获取API URL
    print("请输入B站评论API请求地址（例如：/x/v2/reply/wbi/main?oid=965370992&...）：")
    api_url = input().strip()
    
    # 获取Cookie
    print("\n请输入Cookie（可选，如需登录才能查看的评论则必填）：")
    cookies_str = input().strip()
    
    # 获取高级选项
    print("\n是否需要设置高级选项？(y/n)：")
    advanced_options = input().strip().lower() == 'y'
    
    max_pages = 50  # 默认最大页数
    get_sub_comments = True  # 默认获取子评论
    start_page = 1  # 默认从第1页开始
    delay_base = 2  # 默认基础延迟时间（秒）
    use_random_delay = True  # 默认使用随机延迟
    
    if advanced_options:
        try:
            print("\n请设置最大获取页数（默认50，设置为0表示不限制）：")
            user_max_pages = int(input().strip() or "50")
            if user_max_pages > 0:
                max_pages = user_max_pages
            elif user_max_pages == 0:
                max_pages = float('inf')  # 无限制
                
            print("\n是否获取子评论？(y/n，默认y)：")
            get_sub_comments = input().strip().lower() != 'n'
            
            print("\n从第几页开始获取？（默认1）：")
            start_page_input = input().strip()
            if start_page_input:
                start_page = max(1, int(start_page_input))
                
            print("\n请设置请求间隔基础时间（秒，默认2）：")
            delay_input = input().strip()
            if delay_input:
                delay_base = max(0.5, float(delay_input))
                
            print("\n是否使用随机延迟？(y/n，默认y)：")
            use_random_delay = input().strip().lower() != 'n'
            
            # 如果从非第一页开始，需要设置初始分页参数
            if start_page > 1:
                print("\n从非第一页开始需要提供上一页的offset值：")
                offset_value = input("请输入上一页的offset值（在调试信息中可以找到）：").strip()
                if offset_value:
                    # 解析API URL
                    base_url, params = parse_api_url(api_url)
                    # 更新分页参数
                    pagination_str = {
                        "offset": offset_value
                    }
                    params["pagination_str"] = json.dumps(pagination_str)
                    # 重建API URL
                    query_string = urllib.parse.urlencode(params)
                    api_url = f"{base_url}?{query_string}"
                    print(f"已更新API URL，将从offset={offset_value}开始获取")
                    
        except ValueError as e:
            print(f"输入无效，使用默认值: {e}")
    
    # 获取评论
    print("\n开始获取评论，请稍候...")
    print(f"配置信息: 最大页数={max_pages}, 获取子评论={get_sub_comments}, 起始页={start_page}")
    print(f"延迟设置: 基础延迟={delay_base}秒, 随机延迟={use_random_delay}")
    
    # 修改get_all_comments函数调用，传递更多参数
    start_time = time.time()
    
    # 定义一个内部函数来修改全局变量
    def modify_delay_settings():
        global random
        # 保存原始的random.uniform函数
        original_uniform = random.uniform
        
        # 如果不使用随机延迟，替换random.uniform为固定值函数
        if not use_random_delay:
            def fixed_delay(min_val, max_val):
                # 使用最小值作为固定延迟
                return min_val
            random.uniform = fixed_delay
        
        # 修改延迟基础值
        def get_delay_multiplier(base_delay):
            return base_delay * delay_base / 2  # 调整为用户设置的基础延迟
        
        # 保存原始函数
        original_sleep = time.sleep
        
        # 替换time.sleep函数
        def custom_sleep(seconds):
            # 应用延迟倍数
            adjusted_seconds = get_delay_multiplier(seconds)
            return original_sleep(adjusted_seconds)
        
        # 替换函数
        time.sleep = custom_sleep
        
        return original_uniform, original_sleep
    
    # 修改延迟设置
    original_uniform, original_sleep = modify_delay_settings()
    
    try:
        comments = get_all_comments(api_url, cookies_str, max_pages, get_sub_comments)
    finally:
        # 恢复原始函数
        random.uniform = original_uniform
        time.sleep = original_sleep
    
    end_time = time.time()
    
    # 统计信息
    total_comments = len(comments)
    total_sub_comments = sum(len(comment.get("子评论", [])) for comment in comments)
    
   
    print(f"评论获取完成，共耗时 {end_time - start_time:.2f} 秒")
    print(f"主评论数量: {total_comments}")
    if get_sub_comments:
        print(f"子评论数量: {total_sub_comments}")
    print(f"总评论数量: {total_comments + total_sub_comments}")
    print("=" * 50)
    
    # 打印前5条评论预览
    if comments:
        print("\n评论预览（前5条）：")
        for i, comment in enumerate(comments[:5], 1):
            print(f"\n评论 {i}:")
            print(f"用户：{comment['用户名']} (ID: {comment['用户ID']})")
            print(f"内容：{comment['内容']}")
            print(f"点赞数：{comment['点赞数']}")
            print(f"发布时间：{comment['发布时间']}")
            print(f"IP属地：{comment['IP属地']}")
            print(f"回复数：{comment['回复数']}")
    
    # 询问是否保存文件
    if comments:
                    print("\n请选择保存格式：")
                    print("1. JSON格式（保留完整数据结构）")
                    print("2. CSV格式（便于在Excel中查看）")
                    print("3. 两种格式都保存")
                    print("0. 不保存")
            
                    save_choice = input("请输入选择（0-3）：").strip()
            
                    if save_choice in ['1', '2', '3']:
                        # 保存JSON
                        if save_choice in ['1', '3']:
                            filename = input("请输入JSON文件名（默认为comments.json）：").strip() or "comments.json"
                            if not filename.endswith('.json'):
                                filename += '.json'
                    
                            with open(filename, "w", encoding="utf-8") as f:
                                json.dump(comments, f, ensure_ascii=False, indent=2)
                    
                            print(f"评论已保存到 {filename}")
                
                        # 保存CSV
                        if save_choice in ['2', '3']:
                            filename = input("请输入CSV文件名（默认为comments.csv）：").strip() or "comments.csv"
                            if not filename.endswith('.csv'):
                                filename += '.csv'
                    
                            # 准备CSV数据
                            csv_data = []
                            for comment in comments:
                                # 添加主评论
                                row = {
                                    '评论类型': '主评论',
                                    '用户名': comment['用户名'],
                                    '用户ID': comment['用户ID'],
                                    '内容': comment['内容'],
                                    '点赞数': comment['点赞数'],
                                    '发布时间': comment['发布时间'],
                                    'IP属地': comment['IP属地'],
                                    '回复数': comment['回复数']
                                }
                                csv_data.append(row)
                        
                                # 添加子评论
                                for sub_comment in comment.get('子评论', []):
                                    row = {
                                        '评论类型': '子评论',
                                        '用户名': sub_comment['用户名'],
                                        '用户ID': sub_comment['用户ID'],
                                        '内容': sub_comment['内容'],
                                        '点赞数': sub_comment['点赞数'],
                                        '发布时间': sub_comment['发布时间'],
                                        'IP属地': sub_comment['IP属地'],
                                        '回复数': 0
                                    }
                                    csv_data.append(row)
                    
                            # 使用pandas保存CSV
                            df = pd.DataFrame(csv_data)
                            df.to_csv(filename, index=False, encoding='utf-8-sig')
                            print(f"评论已保存到 {filename}")