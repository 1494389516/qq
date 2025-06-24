import requests
import json
from bs4 import BeautifulSoup
import time


url="https://baidu.com"
headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive"
}
params={
    "key": "value",  # 根据需要添加请求参数
    "key2": "value2"
}

t=requests.get(url, headers=headers, params=params)
if t.status_code == 200:
    soup = BeautifulSoup(t.text, 'html.parser')
    title = soup.title.string if soup.title else 'No title found'
    print(f"提取到的所有链接")
    for link in soup.find_all('a', href=True):
        print(link['href'])
 

      