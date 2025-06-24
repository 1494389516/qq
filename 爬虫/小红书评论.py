import json
import requests
import argparse
import time
import sys
import urllib.parse

def parse_api_path(path):
    """从API路径中提取参数
    
    参数:
        path: API请求路径，例如：/api/sns/web/v2/comment/page?note_id=xxx&cursor=&xsec_token=xxx
        
    返回:
        包含提取参数的字典
    """
    # 提取查询参数部分
    if '?' in path:
        query_string = path.split('?', 1)[1]
    else:
        return {}
        
    # 解析查询参数
    params = {}
    for param in query_string.split('&'):
        if '=' in param:
            key, value = param.split('=', 1)
            params[key] = urllib.parse.unquote(value)
            
    return params

def get_comments(note_id, cookie, xsec_token, cursor="", page_count=None, all_pages=False, request_interval=1, max_retries=3):
    """获取小红书笔记的评论
    
    参数:
        note_id: 笔记ID
        cookie: 用户的完整cookie字符串
        xsec_token: URL参数中的xsec_token值
        cursor: 分页游标，默认为空字符串（第一页）
        page_count: 要获取的页数，如果为None且all_pages为True，则获取所有页
        all_pages: 是否获取所有页面的评论
        request_interval: 请求间隔时间(秒)，避免请求过于频繁
        max_retries: 请求失败时的最大重试次数
    
    返回:
        所有获取到的评论列表
    """
    url = "https://edith.xiaohongshu.com/api/sns/web/v2/comment/page"
    
    headers = {
        "Host": "edith.xiaohongshu.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cookie": cookie,
        "Referer": "https://www.xiaohongshu.com/",
        "Origin": "https://www.xiaohongshu.com"
    }
    
    all_comments = []
    page = 1
    has_more = True
    
    while has_more:
        if page_count is not None and page > page_count:
            break
            
        # 构建请求参数
        params = {
            "note_id": note_id,
            "cursor": cursor,
            "top_comment_id": "",
            "image_formats": "jpg,webp,avif",
            "xsec_token": xsec_token
        }
        
        try:
            print(f"正在获取第 {page} 页评论...")
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and "comments" in data["data"]:
                    comments = data["data"]["comments"]
                    all_comments.extend(comments)
                    print(f"成功获取 {len(comments)} 条评论")
                    
                    # 检查是否有更多页
                    has_more = data["data"].get("has_more", False)
                    cursor = data["data"].get("cursor", "")
                    
                    if not has_more or not all_pages:
                        break
                        
                    # 添加请求间隔
                    time.sleep(request_interval)
                else:
                    print("未找到评论数据")
                    break
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print("响应内容:")
                print(response.text[:500])  # 只打印前500个字符
                break
                
        except Exception as e:
            print(f"请求异常: {e}")
            break
            
        page += 1
        
    return all_comments

def save_comments_to_file(comments, filename):
    """将评论保存到文件
    
    参数:
        comments: 评论列表
        filename: 文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(comments, f, ensure_ascii=False, indent=4)
    print(f"评论已保存到文件: {filename}")

def print_comments(comments, limit=None):
    """打印评论
    
    参数:
        comments: 评论列表
        limit: 最多打印的评论数，如果为None则打印所有
    """
    if limit is not None:
        comments_to_print = comments[:limit]
    else:
        comments_to_print = comments
        
    print(f"\n共获取到 {len(comments)} 条评论")
    
    for i, comment in enumerate(comments_to_print, 1):
        print(f"\n评论 {i}:")
        if "content" in comment:
            print(f"内容: {comment['content']}")
        if "user" in comment and "nickname" in comment["user"]:
            print(f"用户: {comment['user']['nickname']}")
        if "time" in comment:
            print(f"时间: {comment['time']}")
        if "liked_count" in comment:
            print(f"点赞数: {comment['liked_count']}")

def main():
    parser = argparse.ArgumentParser(description='小红书笔记评论爬虫')
    parser.add_argument('--path', type=str, help='完整的API请求路径')
    parser.add_argument('--note_id', type=str, help='笔记ID')
    parser.add_argument('--cookie', type=str, help='完整的cookie字符串')
    parser.add_argument('--token', type=str, help='URL参数中的xsec_token值')
    parser.add_argument('--pages', type=int, help='要获取的页数')
    parser.add_argument('--first-page-only', action='store_true', help='只获取第一页评论')
    parser.add_argument('--all', action='store_true', help='获取所有页面的评论（默认行为）')
    parser.add_argument('--save', type=str, help='保存评论到指定文件')
    parser.add_argument('--interval', type=float, default=1, help='请求间隔时间(秒)')
    
    args = parser.parse_args()
    
    # 如果没有提供命令行参数，则使用交互式输入
    note_id = args.note_id
    cookie = args.cookie
    xsec_token = args.token
    api_path = args.path
    
    if api_path is None and (note_id is None or xsec_token is None):
        print("请选择输入方式：")
        print("1. 输入完整的API请求路径")
        print("2. 分别输入笔记ID和xsec_token")
        choice = input("请输入选择（1或2）: ")
        
        if choice == "1":
            api_path = input("请输入完整的API请求路径: ")
            params = parse_api_path(api_path)
            note_id = params.get('note_id', '')
            xsec_token = params.get('xsec_token', '')
        else:
            note_id = input("请输入笔记ID: ")
            xsec_token = input("请输入URL参数中的xsec_token值: ")
    elif api_path is not None:
        params = parse_api_path(api_path)
        note_id = params.get('note_id', note_id)
        xsec_token = params.get('xsec_token', xsec_token)
        
    if cookie is None:
        cookie = input("请输入完整的cookie字符串: ")
    
    # 验证必要参数
    if not note_id:
        print("错误：未提供笔记ID，请检查API路径或直接输入笔记ID")
        return
        
    if not xsec_token:
        print("错误：未提供xsec_token，请检查API路径或直接输入xsec_token")
        return
        
    if not cookie:
        print("错误：未提供cookie，请输入完整的cookie字符串")
        return
        
    print(f"\n=== 评论获取配置 ===")
    print(f"笔记ID: {note_id}")
    print(f"xsec_token: {xsec_token[:10]}...（已截断）")
    print(f"Cookie长度: {len(cookie)} 字符")
    
    # 确定是否获取所有页面
    # 默认获取所有页面，除非指定了--first-page-only或设置了具体页数
    get_all_pages = not args.first_page_only
    
    # 显示获取模式
    if args.first_page_only:
        print("获取模式: 仅获取第一页评论")
    elif args.pages:
        print(f"获取模式: 获取指定页数（{args.pages}页）")
    else:
        print("获取模式: 获取所有页面评论（默认模式）")
    
    print(f"请求间隔: {args.interval}秒")
    print("=" * 20)
    
    # 获取评论
    comments = get_comments(
        note_id=note_id,
        cookie=cookie,
        xsec_token=xsec_token,
        page_count=args.pages,
        all_pages=get_all_pages,
        request_interval=args.interval
    )
    
    # 打印评论
    print_comments(comments)
    
    # 保存评论到文件
    if args.save:
        save_comments_to_file(comments, args.save)
    elif len(comments) > 0:
        save_option = input("\n是否保存评论到文件？(y/n): ")
        if save_option.lower() == 'y':
            filename = input("请输入文件名: ")
            if filename:
                save_comments_to_file(comments, filename)
    
if __name__ == "__main__":
    main()