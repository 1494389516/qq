#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyInstaller钩子脚本
解决打包时可能遇到的依赖问题
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 确保undetected_chromedriver正确打包
def hook_undetected_chromedriver(hook_api):
    hook_api.add_imports('undetected_chromedriver')
    hook_api.add_imports('undetected_chromedriver.patcher')
    hook_api.add_imports('undetected_chromedriver.options')
    hook_api.add_imports('undetected_chromedriver.cdp')
    hook_api.add_imports('undetected_chromedriver.v2')

# 确保selenium正确打包
def hook_selenium(hook_api):
    hook_api.add_imports('selenium')
    hook_api.add_imports('selenium.webdriver')
    hook_api.add_imports('selenium.webdriver.chrome')
    hook_api.add_imports('selenium.webdriver.chrome.options')
    hook_api.add_imports('selenium.webdriver.chrome.service')
    hook_api.add_imports('selenium.webdriver.common.by')
    hook_api.add_imports('selenium.webdriver.support.ui')
    hook_api.add_imports('selenium.webdriver.support.expected_conditions')
    hook_api.add_imports('selenium.common.exceptions')

# 确保PIL正确打包
def hook_pil(hook_api):
    hook_api.add_imports('PIL')
    hook_api.add_imports('PIL.Image')
    hook_api.add_imports('PIL.ImageTk')
    hook_api.add_imports('PIL._tkinter_finder')

# 确保colorama正确打包
def hook_colorama(hook_api):
    hook_api.add_imports('colorama')
    hook_api.add_imports('colorama.initialise')
    hook_api.add_imports('colorama.ansi')
    hook_api.add_imports('colorama.ansitowin32')

# 确保fake_useragent正确打包
def hook_fake_useragent(hook_api):
    hook_api.add_imports('fake_useragent')
    hook_api.add_imports('fake_useragent.utils')
    hook_api.add_imports('fake_useragent.settings')
    datas = collect_data_files('fake_useragent')
    hook_api.add_datas(datas)

# 确保tqdm正确打包
def hook_tqdm(hook_api):
    hook_api.add_imports('tqdm')
    hook_api.add_imports('tqdm.auto')
    hook_api.add_imports('tqdm.notebook')
    hook_api.add_imports('tqdm.std')

# 确保scrapy正确打包
def hook_scrapy(hook_api):
    hook_api.add_imports('scrapy')
    hook_api.add_imports('scrapy.crawler')
    hook_api.add_imports('scrapy.utils.project')
    hook_api.add_imports('scrapy.exporters')
    hook_api.add_imports('scrapy.pipelines.images')
    hook_api.add_imports('scrapy.signals')

# 确保pyautogui正确打包
def hook_pyautogui(hook_api):
    hook_api.add_imports('pyautogui')
    hook_api.add_imports('pyautogui._pyautogui_win')
    hook_api.add_imports('pyautogui._pyautogui_x11')
    hook_api.add_imports('pyautogui._pyautogui_osx') 