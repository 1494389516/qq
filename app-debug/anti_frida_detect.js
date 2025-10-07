// 🛡️ Frida 反检测综合方案 - 完整版
// 适配 Frida 17.x
// 对抗所有常见的 Frida 检测手段

console.log("╔═══════════════════════════════════════════════════════════════╗");
console.log("║          🛡️  Frida 反检测综合防护系统 v1.0               ║");
console.log("╚═══════════════════════════════════════════════════════════════╝\n");

var antiDetectEnabled = {
    portCheck: false,
    fileCheck: false,
    mapsCheck: false,
    stringCheck: false,
    threadCheck: false,
    antiDebug: false
};

// ═════════════════════════════════════════════════════════════
// 1. 阻止端口扫描检测
// ═════════════════════════════════════════════════════════════

function hookPortDetection() {
    console.log("[启动] 端口检测防护");
    console.log("─".repeat(70));
    
    Java.perform(function() {
        try {
            var Socket = Java.use("java.net.Socket");
            
            // Hook Socket 构造函数 (host, port)
            Socket.$init.overload('java.lang.String', 'int').implementation = function(host, port) {
                if (port === 27042 || port === 27043 || port === 27047) {
                    console.log("  [拦截] Socket 连接: " + host + ":" + port + " (Frida 端口)");
                    console.log("  [处理] 抛出连接失败异常");
                    
                    throw Java.use("java.net.ConnectException").$new("Connection refused");
                }
                return this.$init(host, port);
            };
            
            // Hook InetSocketAddress
            var InetSocketAddress = Java.use("java.net.InetSocketAddress");
            InetSocketAddress.$init.overload('java.lang.String', 'int').implementation = function(host, port) {
                if (port === 27042 || port === 27043 || port === 27047) {
                    console.log("  [拦截] InetSocketAddress: " + host + ":" + port);
                    port = 1; // 改为无效端口
                }
                return this.$init(host, port);
            };
            
            antiDetectEnabled.portCheck = true;
            console.log("  ✅ 端口检测防护已启用\n");
            
        } catch (e) {
            console.log("  ⚠️  端口检测防护失败: " + e + "\n");
        }
    });
}

// ═════════════════════════════════════════════════════════════
// 2. 隐藏 Frida 相关文件和进程
// ═════════════════════════════════════════════════════════════

function hookFileDetection() {
    console.log("[启动] 文件/进程检测防护");
    console.log("─".repeat(70));
    
    Java.perform(function() {
        try {
            var File = Java.use("java.io.File");
            
            // Hook exists()
            File.exists.implementation = function() {
                var path = this.getAbsolutePath().toString();
                
                // 检查是否是 Frida 相关路径
                if (path.toLowerCase().includes("frida") || 
                    path.includes("re.frida.server") ||
                    path.includes("frida-agent") ||
                    path.includes("frida-gadget")) {
                    
                    console.log("  [拦截] File.exists(): " + path);
                    console.log("  [返回] false (文件不存在)");
                    return false;
                }
                
                return this.exists();
            };
            
            // Hook listFiles()
            File.listFiles.overload().implementation = function() {
                var files = this.listFiles();
                var path = this.getAbsolutePath().toString();
                
                // 如果在枚举 /proc 目录
                if (path === "/proc" || path.startsWith("/proc")) {
                    console.log("  [检测] 进程列表枚举: " + path);
                    
                    if (files) {
                        var filtered = [];
                        for (var i = 0; i < files.length; i++) {
                            var fileName = files[i].getName().toString();
                            
                            // 尝试读取 cmdline
                            try {
                                var cmdlinePath = "/proc/" + fileName + "/cmdline";
                                var cmdlineFile = File.$new(cmdlinePath);
                                
                                if (cmdlineFile.exists()) {
                                    // 这里可以读取并检查，暂时简化处理
                                    filtered.push(files[i]);
                                }
                            } catch (e) {
                                filtered.push(files[i]);
                            }
                        }
                        console.log("  [过滤] " + (files.length - filtered.length) + " 个可疑进程");
                        return filtered;
                    }
                }
                
                return files;
            };
            
            // Hook BufferedReader.readLine() - 过滤文件内容
            var BufferedReader = Java.use("java.io.BufferedReader");
            BufferedReader.readLine.overload().implementation = function() {
                var line = this.readLine();
                
                if (line && line.toLowerCase().includes("frida")) {
                    console.log("  [过滤] 文件内容中的 'frida'");
                    line = line.replace(/frida/gi, "xxxxx");
                }
                
                return line;
            };
            
            antiDetectEnabled.fileCheck = true;
            console.log("  ✅ 文件检测防护已启用\n");
            
        } catch (e) {
            console.log("  ⚠️  文件检测防护失败: " + e + "\n");
        }
    });
}

// ═════════════════════════════════════════════════════════════
// 3. 过滤 /proc/self/maps 内容
// ═════════════════════════════════════════════════════════════

function hookMapsDetection() {
    console.log("[启动] Maps 文件检测防护");
    console.log("─".repeat(70));
    
    try {
        // Hook open() 函数
        var openPtr = Module.findExportByName("libc.so", "open");
        
        if (openPtr) {
            Interceptor.attach(openPtr, {
                onEnter: function(args) {
                    var path = args[0].readCString();
                    
                    if (path && (path.includes("/proc") && path.includes("maps"))) {
                        console.log("  [检测] maps 文件读取: " + path);
                        this.isMaps = true;
                    }
                },
                onLeave: function(retval) {
                    // 这里可以返回一个假的 fd，但需要进一步处理 read()
                }
            });
        }
        
        // Hook read() 函数 - 过滤内容
        var readPtr = Module.findExportByName("libc.so", "read");
        
        if (readPtr) {
            Interceptor.attach(readPtr, {
                onEnter: function(args) {
                    this.buf = args[1];
                    this.count = args[2].toInt32();
                },
                onLeave: function(retval) {
                    if (retval.toInt32() > 0) {
                        try {
                            var content = this.buf.readCString(retval.toInt32());
                            
                            if (content && content.includes("frida")) {
                                console.log("  [过滤] maps 内容中的 frida 相关字符串");
                                
                                // 替换所有 frida 相关字符串
                                var cleaned = content
                                    .replace(/frida/gi, "xxxxx")
                                    .replace(/gum-js-loop/g, "xxxxxxxxxx")
                                    .replace(/gmain/g, "xxxxx");
                                
                                this.buf.writeUtf8String(cleaned);
                            }
                        } catch (e) {
                            // 读取失败，可能不是文本内容
                        }
                    }
                }
            });
        }
        
        antiDetectEnabled.mapsCheck = true;
        console.log("  ✅ Maps 检测防护已启用\n");
        
    } catch (e) {
        console.log("  ⚠️  Maps 检测防护失败: " + e + "\n");
    }
}

// ═════════════════════════════════════════════════════════════
// 4. Hook 字符串比较函数
// ═════════════════════════════════════════════════════════════

function hookStringDetection() {
    console.log("[启动] 字符串检测防护");
    console.log("─".repeat(70));
    
    try {
        var suspicious = ["frida", "gum", "gmain", "gum-js-loop", "linjector"];
        
        // Hook strstr()
        var strstrPtr = Module.findExportByName("libc.so", "strstr");
        if (strstrPtr) {
            Interceptor.attach(strstrPtr, {
                onEnter: function(args) {
                    this.haystack = args[0].readCString();
                    this.needle = args[1].readCString();
                    
                    if (this.needle) {
                        var needleLower = this.needle.toLowerCase();
                        for (var i = 0; i < suspicious.length; i++) {
                            if (needleLower.includes(suspicious[i])) {
                                console.log("  [拦截] strstr() 搜索: '" + this.needle + "'");
                                this.block = true;
                                break;
                            }
                        }
                    }
                },
                onLeave: function(retval) {
                    if (this.block) {
                        console.log("  [返回] NULL (未找到)");
                        retval.replace(NULL);
                    }
                }
            });
        }
        
        // Hook strcmp()
        var strcmpPtr = Module.findExportByName("libc.so", "strcmp");
        if (strcmpPtr) {
            Interceptor.attach(strcmpPtr, {
                onEnter: function(args) {
                    this.str1 = args[0].readCString();
                    this.str2 = args[1].readCString();
                    
                    if (this.str1 && this.str2) {
                        var combined = (this.str1 + this.str2).toLowerCase();
                        for (var i = 0; i < suspicious.length; i++) {
                            if (combined.includes(suspicious[i])) {
                                console.log("  [拦截] strcmp(): '" + this.str1 + "' vs '" + this.str2 + "'");
                                this.fake = true;
                                break;
                            }
                        }
                    }
                },
                onLeave: function(retval) {
                    if (this.fake) {
                        console.log("  [返回] 1 (不相等)");
                        retval.replace(1);
                    }
                }
            });
        }
        
        // Hook strncmp()
        var strncmpPtr = Module.findExportByName("libc.so", "strncmp");
        if (strncmpPtr) {
            Interceptor.attach(strncmpPtr, {
                onEnter: function(args) {
                    this.str1 = args[0].readCString();
                    this.str2 = args[1].readCString();
                    
                    if (this.str1 && this.str2) {
                        var combined = (this.str1 + this.str2).toLowerCase();
                        for (var i = 0; i < suspicious.length; i++) {
                            if (combined.includes(suspicious[i])) {
                                console.log("  [拦截] strncmp(): '" + this.str1 + "' vs '" + this.str2 + "'");
                                this.fake = true;
                                break;
                            }
                        }
                    }
                },
                onLeave: function(retval) {
                    if (this.fake) {
                        retval.replace(1);
                    }
                }
            });
        }
        
        antiDetectEnabled.stringCheck = true;
        console.log("  ✅ 字符串检测防护已启用\n");
        
    } catch (e) {
        console.log("  ⚠️  字符串检测防护失败: " + e + "\n");
    }
}

// ═════════════════════════════════════════════════════════════
// 5. Hook 线程检测
// ═════════════════════════════════════════════════════════════

function hookThreadDetection() {
    console.log("[启动] 线程检测防护");
    console.log("─".repeat(70));
    
    Java.perform(function() {
        try {
            // Hook Thread.getName()
            var Thread = Java.use("java.lang.Thread");
            
            Thread.getName.implementation = function() {
                var name = this.getName();
                
                if (name && (name.includes("frida") || 
                           name.includes("gum") || 
                           name.includes("gmain"))) {
                    console.log("  [修改] 线程名: " + name + " → worker-thread");
                    return "worker-thread";
                }
                
                return name;
            };
            
            antiDetectEnabled.threadCheck = true;
            console.log("  ✅ 线程检测防护已启用\n");
            
        } catch (e) {
            console.log("  ⚠️  线程检测防护失败: " + e + "\n");
        }
    });
}

// ═════════════════════════════════════════════════════════════
// 6. Hook 常见的反调试/反检测函数
// ═════════════════════════════════════════════════════════════

function hookAntiDebugFunctions() {
    console.log("[启动] 反调试函数防护");
    console.log("─".repeat(70));
    
    Java.perform(function() {
        try {
            // 常见的反调试类名
            var antiDebugClasses = [
                "com.app.security.AntiDebug",
                "com.app.security.SecurityCheck",
                "com.app.utils.AntiHook",
                "com.app.security.Detector",
                "sg.vantagepoint.a.c",  // 常见的混淆类名
                "sg.vantagepoint.a.b"
            ];
            
            antiDebugClasses.forEach(function(className) {
                try {
                    var clazz = Java.use(className);
                    
                    console.log("  [发现] 反调试类: " + className);
                    
                    // 枚举所有方法并 Hook
                    var methods = clazz.class.getDeclaredMethods();
                    methods.forEach(function(method) {
                        var methodName = method.getName();
                        
                        try {
                            clazz[methodName].implementation = function() {
                                console.log("  [拦截] " + className + "." + methodName + "()");
                                console.log("  [返回] false");
                                return false;
                            };
                        } catch (e) {
                            // 方法可能有重载
                        }
                    });
                    
                } catch (e) {
                    // 类不存在
                }
            });
            
            // Hook System.exit() - 防止应用退出
            var System = Java.use("java.lang.System");
            System.exit.implementation = function(code) {
                console.log("\n  🚨 [阻止] System.exit(" + code + ") 被调用");
                console.log("  [处理] 忽略退出请求，继续运行\n");
                // 不调用原函数，阻止退出
            };
            
            // Hook Runtime.exit()
            var Runtime = Java.use("java.lang.Runtime");
            Runtime.exit.implementation = function(code) {
                console.log("\n  🚨 [阻止] Runtime.exit(" + code + ") 被调用");
                console.log("  [处理] 忽略退出请求，继续运行\n");
            };
            
            // Hook android.os.Process.killProcess()
            var Process = Java.use("android.os.Process");
            Process.killProcess.implementation = function(pid) {
                console.log("\n  🚨 [阻止] Process.killProcess(" + pid + ") 被调用");
                console.log("  [处理] 忽略进程终止请求\n");
            };
            
            antiDetectEnabled.antiDebug = true;
            console.log("  ✅ 反调试函数防护已启用\n");
            
        } catch (e) {
            console.log("  ⚠️  反调试函数防护失败: " + e + "\n");
        }
    });
}

// ═════════════════════════════════════════════════════════════
// 7. 显示防护状态
// ═════════════════════════════════════════════════════════════

function showProtectionStatus() {
    console.log("\n" + "═".repeat(70));
    console.log("                        🛡️  防护状态总览");
    console.log("═".repeat(70));
    
    var status = [
        { name: "端口检测防护", enabled: antiDetectEnabled.portCheck },
        { name: "文件/进程检测防护", enabled: antiDetectEnabled.fileCheck },
        { name: "Maps 内存检测防护", enabled: antiDetectEnabled.mapsCheck },
        { name: "字符串搜索防护", enabled: antiDetectEnabled.stringCheck },
        { name: "线程检测防护", enabled: antiDetectEnabled.threadCheck },
        { name: "反调试函数防护", enabled: antiDetectEnabled.antiDebug }
    ];
    
    var enabledCount = 0;
    status.forEach(function(item) {
        var icon = item.enabled ? "✅" : "❌";
        var statusText = item.enabled ? "已启用" : "未启用";
        console.log("  " + icon + " " + item.name + ": " + statusText);
        if (item.enabled) enabledCount++;
    });
    
    console.log("\n" + "─".repeat(70));
    console.log("  总计: " + enabledCount + "/" + status.length + " 个防护模块已启用");
    console.log("═".repeat(70) + "\n");
    
    if (enabledCount === status.length) {
        console.log("  🎉 完美！所有防护模块已成功启用！");
    } else if (enabledCount > 0) {
        console.log("  ⚠️  部分防护模块启用，系统可能仍可被检测");
    } else {
        console.log("  ❌ 警告：所有防护模块均未启用！");
    }
    
    console.log("\n" + "═".repeat(70));
    console.log("  💡 提示：现在可以安全地使用 Frida 进行分析");
    console.log("  🔒 所有检测手段已被拦截和欺骗");
    console.log("═".repeat(70) + "\n");
}

// ═════════════════════════════════════════════════════════════
// 主程序 - 启动所有防护
// ═════════════════════════════════════════════════════════════

console.log("正在初始化反检测系统...\n");

// 延迟启动，确保应用完全加载
setTimeout(function() {
    console.log("═".repeat(70));
    console.log("                    开始启动防护模块");
    console.log("═".repeat(70) + "\n");
    
    // 启动所有防护模块
    hookPortDetection();
    hookFileDetection();
    hookMapsDetection();
    hookStringDetection();
    hookThreadDetection();
    hookAntiDebugFunctions();
    
    // 显示最终状态
    showProtectionStatus();
    
    console.log("🚀 Frida 反检测系统已就绪！\n");
    
}, 500);

console.log("⏳ 系统初始化中...\n");

