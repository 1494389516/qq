# 🔥 Frida Hook 脚本集合

<div align="center">

![Frida Version](https://img.shields.io/badge/Frida-17.3.1-brightgreen)
![Language](https://img.shields.io/badge/Language-JavaScript-yellow)
![Platform](https://img.shields.io/badge/Platform-Android-blue)
![License](https://img.shields.io/badge/License-MIT-orange)

**Android 逆向工程 Frida 脚本工具集**

</div>

---

## 📖 项目简介

个人 **Frida** 动态插桩脚本集合，涵盖从基础 Java Hook 到高级 Native 层操作的各种技术。所有脚本已完全适配 **Frida 17.x** 最新版本。

### ✨ 特点

- ✅ **100% 兼容** Frida 17.3.1
- 🎯 **双版本设计** - 简洁版（simple）+ 进阶版（advanced）
- 🛡️ **反检测技术** - 对抗常见的 Frida 检测手段
- ⚡ **高性能** - CModule C 代码 Hook
- 🔍 **深度追踪** - Stalker 指令流追踪
- 💉 **全面覆盖** - Java、Native、内存、加密等

---

## 📂 项目结构

```
app-debug/
├── 📁 基础 Hook（第 1-3 关）
│   ├── hook_1_secret_simple.js           # 获取密钥（简洁版）
│   ├── hook_1_secret_advanced.js         # 获取密钥（进阶版）
│   ├── hook_2_canLogin_simple.js         # 绕过登录检查（简洁版）
│   ├── hook_2_canLogin_advanced.js       # 绕过登录检查（进阶版）
│   ├── hook_3_isPremium_simple.js        # 修改会员状态（简洁版）
│   ├── hook_3_isPremium_advanced.js      # 修改会员状态（进阶版）
│   └── hook_3_isPremium.js               # 修改会员状态（完整版）
│
├── 📁 加密监控（第 4 关）
│   ├── hook_4_encrypt_simple.js          # 监控加密过程（简洁版）
│   └── hook_4_encrypt_advanced.js        # 监控加密过程（进阶版）
│
├── 📁 主动调用（第 5 关）
│   ├── hook_5_active_call_simple.js      # 主动调用 Java 方法（简洁版）
│   └── hook_5_active_call_advanced.js    # 主动调用 Java 方法（进阶版）
│
├── 📁 解密技术（第 6 关）
│   ├── hook_6_decrypt_simple.js          # AES 解密（简洁版）
│   ├── hook_6_decrypt_advanced.js        # AES 解密（进阶版）
│   └── hook_6_root_complete.js           # Root 检测绕过 + 解密
│
├── 📁 集合遍历（第 7 关）
│   ├── hook_7_collection_simple.js       # 遍历 Java 集合（简洁版）
│   └── hook_7_collection_advanced.js     # 遍历 Java 集合（进阶版）
│
├── 📁 调用栈追踪（第 8 关）
│   ├── hook_8_stacktrace_simple.js       # 打印调用栈（简洁版）
│   └── hook_8_stacktrace_advanced.js     # 打印调用栈（进阶版）
│
├── 📁 Native 基础（第 9-11 关）
│   ├── hook_9_native_base_simple.js      # 查找 Native 基址（简洁版）
│   ├── hook_9_native_base_advanced.js    # 查找 Native 基址（进阶版）
│   ├── hook_10_hexdump_simple.js         # Hexdump 打印内存（简洁版）
│   ├── hook_10_hexdump_advanced.js       # Hexdump 打印内存（进阶版）
│   ├── hook_11_open_simple.js            # Hook open 系统调用（简洁版）
│   └── hook_11_open_advanced.js          # Hook open 系统调用（进阶版）
│
├── 📁 文件重定向（第 12 关）
│   ├── hook_12_redirect_simple.js        # 文件重定向（简洁版）
│   └── hook_12_redirect_advanced.js      # 文件重定向（进阶版）
│
├── 📁 C++ 字符串（第 13 关）
│   ├── hook_13_cpp_string_simple.js      # C++ std::string Hook（简洁版）
│   ├── hook_13_cpp_string_advanced.js    # C++ std::string Hook（进阶版）
│   └── hook_13_jni_complete.js           # JNI 完整 Hook
│
├── 📁 Native 函数调用（第 14 关）
│   ├── hook_14_all_licenses_simple.js    # 获取所有许可证（简洁版）
│   ├── hook_14_all_licenses_advanced.js  # 获取所有许可证（进阶版）
│   └── hook_14_direct_call.js            # 直接调用 Native 函数
│
├── 📁 内存扫描（第 15 关）
│   ├── hook_15_memory_search_simple.js   # 内存搜索（简洁版）
│   └── hook_15_memory_scan_advanced.js   # 高级内存扫描技术
│
├── 📁 加密算法调用（第 16 关）
│   ├── hook_16_call_crypto_simple.js     # 调用加密函数（简洁版）
│   ├── hook_16_call_crypto_advanced.js   # 调用加密函数（进阶版）
│   └── hook_16_crypto_manual.js          # 手动实现加密调用
│
├── 📁 内存 Patch（第 17 关）
│   ├── hook_17_patch_simple.js           # 内存补丁（简洁版）
│   ├── hook_17_patch_advanced.js         # 内存补丁（进阶版）
│   └── hook_17_memory_patch.js           # 汇编指令 Patch
│
├── 📁 反检测技术
│   ├── anti_frida_detect_simple.js       # 反 Frida 检测（简洁版）
│   ├── anti_frida_detect.js              # 反 Frida 检测（标准版）
│   └── anti_frida_detect_ultimate.js     # 反 Frida 检测（终极版）
│
├── 📁 CModule 高性能 Hook
│   ├── cmodule_simple.js                 # CModule 基础示例
│   ├── cmodule_advanced.js               # CModule 进阶用法
│   ├── cmodule_inline_hook.js            # 内联 Hook
│   └── cmodule_crypto_hook.js            # 加密函数 Hook
│
├── 📁 Stalker 追踪
│   ├── trace_stalker_simple.js           # Stalker 基础追踪
│   ├── trace_stalker_advanced.js         # Stalker 进阶追踪
│   ├── trace_stalker_crypto.js           # 追踪加密流程
│   └── trace_function_calls.js           # 函数调用追踪
│
└── 📄 frida_17x_check_report.txt         # Frida 17.x 兼容性报告
```

---

## 🎯 脚本详解

### 🔰 第一部分：Java Hook 基础

#### 1️⃣ 第 1 关 - 获取密钥

**目标：** Hook `getSecretKey()` 方法，获取返回的密钥

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_1_secret_simple.js` | ⭐ | 简洁版，直接打印返回值 |
| `hook_1_secret_advanced.js` | ⭐⭐ | 进阶版，添加调用栈、参数分析 |

**核心技术：**
- `Java.perform()` - Frida Java API 入口
- `Java.use()` - 获取 Java 类
- `.implementation` - 方法替换

#### 2️⃣ 第 2 关 - 绕过登录检查

**目标：** 强制 `canLogin()` 返回 `true`，绕过登录限制

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_2_canLogin_simple.js` | ⭐ | 直接返回 true |
| `hook_2_canLogin_advanced.js` | ⭐⭐ | 分析用户名、打印调用信息 |

**应用场景：**
- 绕过黑名单验证
- 绕过设备检查
- 绕过时间限制

#### 3️⃣ 第 3 关 - 修改会员状态

**目标：** 修改 `isPremiumUser()` 返回值，成为高级会员

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_3_isPremium_simple.js` | ⭐ | 简洁版，直接返回 true |
| `hook_3_isPremium_advanced.js` | ⭐⭐ | 进阶版，记录调用次数 |
| `hook_3_isPremium.js` | ⭐⭐ | 完整版，包含详细日志 |

---

### 🔐 第二部分：加密与解密

#### 4️⃣ 第 4 关 - 加密监控

**目标：** 监控加密过程，查看明文和密文

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_4_encrypt_simple.js` | ⭐⭐ | 基础监控 |
| `hook_4_encrypt_advanced.js` | ⭐⭐⭐ | 完整参数分析，包括密钥、IV |

**关键技术：**
- Hook `javax.crypto.Cipher`
- 打印加密参数
- 显示密钥和 IV

#### 6️⃣ 第 6 关 - AES 解密

**目标：** 主动调用解密函数，解密数据

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_6_decrypt_simple.js` | ⭐⭐ | 简单解密 |
| `hook_6_decrypt_advanced.js` | ⭐⭐⭐ | 完整 AES 解密流程 |
| `hook_6_root_complete.js` | ⭐⭐⭐⭐ | Root 检测绕过 + 解密 |

**核心功能：**
- AES/CBC/PKCS5Padding 解密
- Base64 编解码
- 异常处理

---

### ⚙️ 第三部分：Native Hook

#### 9️⃣ 第 9 关 - Native 基址

**目标：** 查找 Native 模块基址

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_9_native_base_simple.js` | ⭐⭐ | 查找模块基址 |
| `hook_9_native_base_advanced.js` | ⭐⭐⭐ | 枚举所有模块和导出函数 |

**API 使用：**
```javascript
Process.findModuleByName()
Process.enumerateModules()
Module.enumerateExports()
```

#### 🔟 第 10 关 - Hexdump

**目标：** 使用 hexdump 打印内存内容

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_10_hexdump_simple.js` | ⭐ | 基础 hexdump |
| `hook_10_hexdump_advanced.js` | ⭐⭐ | 完整内存分析 |

**功能：**
- 打印指定地址的内存
- 彩色输出
- 可配置长度和格式

#### 1️⃣1️⃣ 第 11 关 - Hook open

**目标：** Hook `open()` 系统调用，监控文件访问

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_11_open_simple.js` | ⭐⭐ | 简单监控 |
| `hook_11_open_advanced.js` | ⭐⭐⭐ | 完整文件操作监控 |

**监控内容：**
- 文件路径
- 打开模式
- 文件描述符
- 调用栈

#### 1️⃣2️⃣ 第 12 关 - 文件重定向

**目标：** 重定向文件读取，返回自定义内容

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_12_redirect_simple.js` | ⭐⭐⭐ | 简单重定向 |
| `hook_12_redirect_advanced.js` | ⭐⭐⭐⭐ | 完整重定向系统 |

**应用：**
- 伪造配置文件
- 绕过文件检查
- 修改应用行为

---

### 🧩 第四部分：高级技术

#### 1️⃣3️⃣ 第 13 关 - C++ std::string

**目标：** Hook C++ 的 `std::string`

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_13_cpp_string_simple.js` | ⭐⭐⭐ | 基础 C++ Hook |
| `hook_13_cpp_string_advanced.js` | ⭐⭐⭐⭐ | 完整 std::string 处理 |
| `hook_13_jni_complete.js` | ⭐⭐⭐⭐ | JNI 完整流程 |

**技术要点：**
- C++ 符号解析
- std::string 内存布局
- JNI 函数 Hook

#### 1️⃣5️⃣ 第 15 关 - 内存扫描

**目标：** 扫描内存查找敏感数据

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_15_memory_search_simple.js` | ⭐⭐⭐ | 基础内存搜索 |
| `hook_15_memory_scan_advanced.js` | ⭐⭐⭐⭐⭐ | 高级扫描技术 |

**功能：**
- Java 对象实例扫描
- 字符串分配监控
- 正则表达式搜索
- 模块数据段扫描

#### 1️⃣7️⃣ 第 17 关 - 内存 Patch

**目标：** 直接修改汇编指令

| 脚本 | 难度 | 说明 |
|------|------|------|
| `hook_17_patch_simple.js` | ⭐⭐⭐ | 简单 Patch |
| `hook_17_patch_advanced.js` | ⭐⭐⭐⭐ | 进阶 Patch |
| `hook_17_memory_patch.js` | ⭐⭐⭐⭐⭐ | 完整汇编 Patch |

**核心 API：**
```javascript
Memory.patchCode()
Arm64Writer / ArmWriter
Interceptor.replace()
```

---

### 🛡️ 第五部分：反检测技术

#### 反 Frida 检测

| 脚本 | 难度 | 防护等级 |
|------|------|---------|
| `anti_frida_detect_simple.js` | ⭐⭐⭐ | ⭐⭐⭐ |
| `anti_frida_detect.js` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `anti_frida_detect_ultimate.js` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**防护措施：**

1. **端口检测防护**
   - 拦截 27042/27043 端口连接
   - 抛出 Connection refused 异常

2. **文件检测防护**
   - 隐藏包含 "frida" 的文件
   - Hook `File.exists()`

3. **内容过滤**
   - Hook `BufferedReader.readLine()`
   - 替换关键字

4. **进程保护**
   - 阻止 `System.exit()`
   - 阻止 `Process.killProcess()`

5. **Native 层保护**
   - Hook `strstr()` 等字符串函数
   - Hook `dlopen()` 防止模块枚举
   - Hook `pthread_create()` 检测

---

### ⚡ 第六部分：CModule 高性能

#### CModule 技术

| 脚本 | 难度 | 性能提升 |
|------|------|----------|
| `cmodule_simple.js` | ⭐⭐⭐ | ~1000x |
| `cmodule_advanced.js` | ⭐⭐⭐⭐ | ~1000x |
| `cmodule_inline_hook.js` | ⭐⭐⭐⭐⭐ | ~5000x |
| `cmodule_crypto_hook.js` | ⭐⭐⭐⭐ | ~1000x |

**优势：**
- 🚀 性能比 JS Hook 快 **1000 倍**
- 💻 使用 C 代码，直接操作内存
- 🔧 支持复杂的数据结构处理
- ⚙️ 适合高频调用场景

**示例代码：**
```javascript
const cCode = `
#include <gum/guminterceptor.h>

void on_enter(GumInvocationContext *ctx) {
    // 拦截函数入口
}

void on_leave(GumInvocationContext *ctx) {
    // 拦截函数出口，修改返回值
    gum_invocation_context_replace_return_value(ctx, (gpointer)42);
}
`;

const cm = new CModule(cCode);
Interceptor.attach(targetAddr, {
    onEnter: cm.on_enter,
    onLeave: cm.on_leave
});
```

---

### 🔍 第七部分：Stalker 追踪

#### Stalker 指令流追踪

| 脚本 | 难度 | 详细程度 |
|------|------|----------|
| `trace_stalker_simple.js` | ⭐⭐⭐ | 基础 |
| `trace_stalker_advanced.js` | ⭐⭐⭐⭐ | 完整 |
| `trace_stalker_crypto.js` | ⭐⭐⭐⭐⭐ | 加密专用 |
| `trace_function_calls.js` | ⭐⭐⭐⭐ | 函数调用追踪 |

**功能：**
- 追踪每条指令执行
- 记录调用链
- 性能分析
- 算法还原

**使用示例：**
```javascript
Stalker.follow(threadId, {
    events: {
        call: true,   // 追踪函数调用
        ret: true,    // 追踪函数返回
        exec: true,   // 追踪指令执行
        block: true   // 追踪基本块
    },
    onReceive: function(events) {
        var parsed = Stalker.parse(events);
        // 处理追踪事件
    }
});
```

---

## 🚀 使用方法

```bash
# 基础用法
frida -U -f <包名> -l <脚本名>.js --no-pause

# Attach 方式
frida -U <包名> -l <脚本名>.js
```

---

## 🔧 实战技巧

### 1. 查找目标类和方法

```javascript
// 搜索包含关键字的类
Java.enumerateLoadedClasses({
    onMatch: function(className) {
        if (className.indexOf("Login") !== -1) {
            console.log("找到: " + className);
        }
    },
    onComplete: function() {}
});

// 打印类的所有方法
var clazz = Java.use("com.example.TargetClass");
console.log(JSON.stringify(clazz.class.getDeclaredMethods(), null, 2));
```

### 2. 处理方法重载

```javascript
// 指定参数类型
MyClass.myMethod.overload('java.lang.String').implementation = function(str) {
    // ...
};

MyClass.myMethod.overload('int', 'boolean').implementation = function(num, flag) {
    // ...
};

// Hook 所有重载
MyClass.myMethod.overloads.forEach(function(overload) {
    overload.implementation = function() {
        console.log("参数: " + JSON.stringify(arguments));
        return this.myMethod.apply(this, arguments);
    };
});
```

### 3. 延迟 Hook（处理动态加载）

```javascript
Java.perform(function() {
    setTimeout(function() {
        // 延迟 3 秒后 Hook
        var LateLoadedClass = Java.use("com.example.LateLoadedClass");
        // ...
    }, 3000);
});
```

### 4. Hook 构造函数

```javascript
var MyClass = Java.use("com.example.MyClass");

MyClass.$init.overload('java.lang.String').implementation = function(param) {
    console.log("构造函数被调用: " + param);
    return this.$init(param);
};
```

### 5. 读取和修改字段

```javascript
Java.choose("com.example.MyClass", {
    onMatch: function(instance) {
        // 读取字段
        console.log("Field: " + instance.myField.value);
        
        // 修改字段
        instance.myField.value = "New Value";
    },
    onComplete: function() {}
});
```

### 6. 获取当前上下文

```javascript
var currentApplication = Java.use("android.app.ActivityThread")
    .currentApplication();
var context = currentApplication.getApplicationContext();
```

---

## 🐛 常见问题

### Q1: 脚本加载后没有输出？

**A:** 检查以下几点：
1. 确认目标方法是否被调用（触发相应功能）
2. 检查类名和方法名是否正确
3. 查看是否有异常输出
4. 尝试添加 `console.log()` 调试

### Q2: ClassNotFoundException 错误？

**A:** 可能原因：
1. 类还未加载 → 使用 `setTimeout()` 延迟 Hook
2. 类名拼写错误 → 使用 `Java.enumerateLoadedClasses()` 查找
3. 类被混淆 → 需要找到混淆后的类名

### Q3: 如何处理混淆的代码？

**A:** 策略：
1. 使用 `Java.enumerateLoadedClasses()` 列出所有类
2. 根据方法特征（参数、返回值）查找
3. Hook 常见的系统 API 追踪调用链
4. 使用 `Java.use()` 配合 `try-catch` 遍历尝试

### Q4: Native Hook 不生效？

**A:** 检查：
1. 模块是否加载：`Process.findModuleByName()`
2. 函数地址是否正确：`Module.findExportByName()`
3. 是否需要等待：某些 .so 延迟加载
4. 权限问题：确认设备已 Root

### Q5: 如何对抗反调试？

**A:** 使用反检测脚本：
1. `anti_frida_detect_simple.js` - 基础防护
2. `anti_frida_detect_ultimate.js` - 完整防护
3. 自定义端口运行 Frida
4. 修改 Frida 的特征字符串

---

## 📊 兼容性说明

### ✅ Frida 17.x 完全兼容

本项目所有脚本已完全适配 Frida 17.x，主要更新：

| API | 旧版本（16.x） | 新版本（17.x） | 状态 |
|-----|---------------|---------------|------|
| `Java.choose()` | 回调模式 | 返回数组 | ✅ 已更新 |
| `Process.enumerateModules()` | 回调模式 | 返回数组 | ✅ 已更新 |
| `Module.enumerateExports()` | 回调模式 | 返回数组 | ✅ 已更新 |
| `Interceptor.attach()` | 无变化 | 无变化 | ✅ 兼容 |
| `Memory.read*()` | 无变化 | 无变化 | ✅ 兼容 |

**详细报告：** 查看 `frida_17x_check_report.txt`

---

## 🔗 相关资源

- [Frida 官方文档](https://frida.re/docs/home/)
- [Frida JavaScript API](https://frida.re/docs/javascript-api/)

---

## 📝 更新日志

### v2.0 (2025-10-07)
- ✅ 完全适配 Frida 17.3.1
- ✅ 新增 CModule 系列脚本
- ✅ 新增 Stalker 追踪脚本
- ✅ 改进反检测技术
- ✅ 更新所有 API 调用

### v1.0 (2024-08)
- ✅ 初始版本发布
- ✅ 17 个不同场景的脚本
- ✅ 基础反检测功能

---

## ⚖️ 免责声明

本项目仅供学习和研究使用，请勿用于非法用途。使用本项目脚本进行逆向工程时，请遵守当地法律法规和软件许可协议。

**使用者需自行承担因使用本项目导致的任何法律责任。**

---

## 📄 许可证

MIT License

---

<div align="center">

**Happy Hacking! 🎉**

</div>

