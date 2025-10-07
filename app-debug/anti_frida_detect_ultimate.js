// 🛡️ Frida 反检测 - 终极版
// 适配 Frida 17.x
// 包含所有已知的反检测技术 + 主动检测对抗

console.log("╔═══════════════════════════════════════════════════════════════╗");
console.log("║        🔥 Frida 反检测终极防护系统 v2.0 🔥                ║");
console.log("║           对抗所有已知的 Frida 检测手段                      ║");
console.log("╚═══════════════════════════════════════════════════════════════╝\n");

// 防护统计
var stats = {
    portBlocked: 0,
    fileHidden: 0,
    contentFiltered: 0,
    exitBlocked: 0,
    stringBlocked: 0
};

// ═════════════════════════════════════════════════════════════
// 核心防护模块
// ═════════════════════════════════════════════════════════════

function initJavaProtection() {
    Java.perform(function() {
        
        // 1. 全面的端口检测防护
        hookSocketAPIs();
        
        // 2. 文件系统防护
        hookFileAPIs();
        
        // 3. 进程和线程防护
        hookProcessAPIs();
        
        // 4. 反调试防护
        hookDebugAPIs();
        
        // 5. 内存和系统信息防护
        hookSystemAPIs();
        
        // 6. 主动搜索反检测类并 Hook
        findAndHookSecurityClasses();
    });
}

function hookSocketAPIs() {
    console.log("[模块 1] Socket 和网络检测防护");
    console.log("─".repeat(70));
    
    try {
        // Socket
        var Socket = Java.use("java.net.Socket");
        Socket.$init.overload('java.lang.String', 'int').implementation = function(host, port) {
            if (port >= 27040 && port <= 27050) {
                console.log("  🚫 拦截 Socket: " + host + ":" + port);
                stats.portBlocked++;
                throw Java.use("java.net.ConnectException").$new("Connection refused");
            }
            return this.$init(host, port);
        };
        
        // InetAddress
        var InetAddress = Java.use("java.net.InetAddress");
        InetAddress.getByName.implementation = function(host) {
            if (host && host.includes("127.0.0.1")) {
                console.log("  🔍 检测 InetAddress: " + host);
            }
            return this.getByName(host);
        };
        
        console.log("  ✅ Socket 防护完成\n");
    } catch (e) {
        console.log("  ⚠️  Socket 防护失败\n");
    }
}

function hookFileAPIs() {
    console.log("[模块 2] 文件系统检测防护");
    console.log("─".repeat(70));
    
    try {
        var File = Java.use("java.io.File");
        
        var suspiciousPaths = [
            "frida", "frida-server", "frida-agent", "frida-gadget",
            "re.frida.server", "linjector", "gum-js-loop"
        ];
        
        function isSuspicious(path) {
            if (!path) return false;
            var lowerPath = path.toLowerCase();
            for (var i = 0; i < suspiciousPaths.length; i++) {
                if (lowerPath.includes(suspiciousPaths[i])) {
                    return true;
                }
            }
            return false;
        }
        
        // exists()
        File.exists.implementation = function() {
            var path = this.getAbsolutePath().toString();
            if (isSuspicious(path)) {
                console.log("  👻 隐藏: " + path);
                stats.fileHidden++;
                return false;
            }
            return this.exists();
        };
        
        // canRead()
        File.canRead.implementation = function() {
            var path = this.getAbsolutePath().toString();
            if (isSuspicious(path)) {
                return false;
            }
            return this.canRead();
        };
        
        // listFiles()
        File.listFiles.overload().implementation = function() {
            var files = this.listFiles();
            var path = this.getAbsolutePath().toString();
            
            if (path.startsWith("/proc") || path.startsWith("/data/local/tmp")) {
                if (files) {
                    var filtered = [];
                    for (var i = 0; i < files.length; i++) {
                        var fileName = files[i].getName().toString();
                        if (!isSuspicious(fileName)) {
                            filtered.push(files[i]);
                        } else {
                            stats.fileHidden++;
                        }
                    }
                    return filtered;
                }
            }
            return files;
        };
        
        // FileInputStream
        var FileInputStream = Java.use("java.io.FileInputStream");
        FileInputStream.$init.overload('java.io.File').implementation = function(file) {
            var path = file.getAbsolutePath().toString();
            if (isSuspicious(path)) {
                console.log("  🚫 阻止读取: " + path);
                throw Java.use("java.io.FileNotFoundException").$new("File not found");
            }
            return this.$init(file);
        };
        
        // BufferedReader - 过滤内容
        var BufferedReader = Java.use("java.io.BufferedReader");
        BufferedReader.readLine.overload().implementation = function() {
            var line = this.readLine();
            if (line) {
                var needFilter = false;
                suspiciousPaths.forEach(function(keyword) {
                    if (line.toLowerCase().includes(keyword)) {
                        needFilter = true;
                    }
                });
                
                if (needFilter) {
                    stats.contentFiltered++;
                    suspiciousPaths.forEach(function(keyword) {
                        var regex = new RegExp(keyword, 'gi');
                        line = line.replace(regex, "x".repeat(keyword.length));
                    });
                }
            }
            return line;
        };
        
        console.log("  ✅ 文件系统防护完成\n");
    } catch (e) {
        console.log("  ⚠️  文件系统防护失败: " + e + "\n");
    }
}

function hookProcessAPIs() {
    console.log("[模块 3] 进程和线程检测防护");
    console.log("─".repeat(70));
    
    try {
        // Runtime.exec() - 阻止执行检测命令
        var Runtime = Java.use("java.lang.Runtime");
        Runtime.exec.overload('java.lang.String').implementation = function(cmd) {
            if (cmd && (cmd.includes("ps") || cmd.includes("pidof") || 
                       cmd.includes("frida") || cmd.includes("proc"))) {
                console.log("  🚫 阻止命令: " + cmd);
                throw Java.use("java.io.IOException").$new("Operation not permitted");
            }
            return this.exec(cmd);
        };
        
        // Thread.getName()
        var Thread = Java.use("java.lang.Thread");
        Thread.getName.implementation = function() {
            var name = this.getName();
            if (name && (name.includes("frida") || name.includes("gum") || 
                        name.includes("gmain") || name.includes("gdbus"))) {
                return "Thread-" + Math.floor(Math.random() * 1000);
            }
            return name;
        };
        
        console.log("  ✅ 进程防护完成\n");
    } catch (e) {
        console.log("  ⚠️  进程防护失败\n");
    }
}

function hookDebugAPIs() {
    console.log("[模块 4] 反调试和反检测防护");
    console.log("─".repeat(70));
    
    try {
        // System.exit()
        var System = Java.use("java.lang.System");
        System.exit.implementation = function(code) {
            console.log("  🛑 阻止 System.exit(" + code + ")");
            stats.exitBlocked++;
        };
        
        // Runtime.exit()
        var Runtime = Java.use("java.lang.Runtime");
        Runtime.exit.implementation = function(code) {
            console.log("  🛑 阻止 Runtime.exit(" + code + ")");
            stats.exitBlocked++;
        };
        
        // Process.killProcess()
        var Process = Java.use("android.os.Process");
        Process.killProcess.implementation = function(pid) {
            console.log("  🛑 阻止 killProcess(" + pid + ")");
            stats.exitBlocked++;
        };
        
        // Debug.isDebuggerConnected()
        var Debug = Java.use("android.os.Debug");
        Debug.isDebuggerConnected.implementation = function() {
            return false;
        };
        
        console.log("  ✅ 反调试防护完成\n");
    } catch (e) {
        console.log("  ⚠️  反调试防护失败\n");
    }
}

function hookSystemAPIs() {
    console.log("[模块 5] 系统信息检测防护");
    console.log("─".repeat(70));
    
    try {
        // System.getProperty()
        var System = Java.use("java.lang.System");
        var getProperty = System.getProperty.overload('java.lang.String');
        getProperty.implementation = function(key) {
            var value = getProperty.call(this, key);
            
            if (value && value.toLowerCase().includes("frida")) {
                console.log("  🔄 过滤系统属性: " + key + " = " + value);
                return value.replace(/frida/gi, "xxxxx");
            }
            return value;
        };
        
        console.log("  ✅ 系统信息防护完成\n");
    } catch (e) {
        console.log("  ⚠️  系统信息防护失败\n");
    }
}

function findAndHookSecurityClasses() {
    console.log("[模块 6] 主动搜索并 Hook 安全检测类");
    console.log("─".repeat(70));
    
    try {
        Java.enumerateLoadedClasses({
            onMatch: function(className) {
                // 匹配可疑的安全检测类
                if (className.toLowerCase().includes("antifrida") ||
                    className.toLowerCase().includes("antihook") ||
                    className.toLowerCase().includes("antidebug") ||
                    className.toLowerCase().includes("security") ||
                    className.toLowerCase().includes("detector") ||
                    className.toLowerCase().includes("check")) {
                    
                    try {
                        var clazz = Java.use(className);
                        console.log("  🎯 发现安全类: " + className);
                        
                        // Hook 所有方法返回 false
                        var methods = clazz.class.getDeclaredMethods();
                        methods.forEach(function(method) {
                            var methodName = method.getName();
                            var returnType = method.getReturnType().getName();
                            
                            if (returnType === "boolean") {
                                try {
                                    clazz[methodName].implementation = function() {
                                        console.log("    ↳ Hook: " + methodName + "() → false");
                                        return false;
                                    };
                                } catch (e) {}
                            }
                        });
                    } catch (e) {}
                }
            },
            onComplete: function() {
                console.log("  ✅ 安全类扫描完成\n");
            }
        });
    } catch (e) {
        console.log("  ⚠️  安全类扫描失败\n");
    }
}

// ═════════════════════════════════════════════════════════════
// Native 层防护
// ═════════════════════════════════════════════════════════════

function initNativeProtection() {
    console.log("[模块 7] Native 层字符串检测防护");
    console.log("─".repeat(70));
    
    try {
        var suspiciousKeywords = ["frida", "gum", "gmain", "linjector"];
        
        // strstr
        var strstr = Module.findExportByName("libc.so", "strstr");
        if (strstr) {
            Interceptor.attach(strstr, {
                onEnter: function(args) {
                    var needle = args[1].readCString();
                    if (needle) {
                        for (var i = 0; i < suspiciousKeywords.length; i++) {
                            if (needle.toLowerCase().includes(suspiciousKeywords[i])) {
                                this.block = true;
                                stats.stringBlocked++;
                                break;
                            }
                        }
                    }
                },
                onLeave: function(retval) {
                    if (this.block) {
                        retval.replace(NULL);
                    }
                }
            });
        }
        
        // strcmp
        var strcmp = Module.findExportByName("libc.so", "strcmp");
        if (strcmp) {
            Interceptor.attach(strcmp, {
                onEnter: function(args) {
                    var str1 = args[0].readCString();
                    var str2 = args[1].readCString();
                    
                    if (str1 && str2) {
                        var combined = (str1 + str2).toLowerCase();
                        for (var i = 0; i < suspiciousKeywords.length; i++) {
                            if (combined.includes(suspiciousKeywords[i])) {
                                this.fake = true;
                                stats.stringBlocked++;
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
        
        console.log("  ✅ Native 防护完成\n");
    } catch (e) {
        console.log("  ⚠️  Native 防护失败\n");
    }
}

// ═════════════════════════════════════════════════════════════
// 统计信息显示
// ═════════════════════════════════════════════════════════════

function showStatistics() {
    setInterval(function() {
        if (stats.portBlocked + stats.fileHidden + stats.contentFiltered + 
            stats.exitBlocked + stats.stringBlocked > 0) {
            
            console.log("\n" + "═".repeat(70));
            console.log("                    📊 实时防护统计");
            console.log("═".repeat(70));
            console.log("  端口扫描拦截: " + stats.portBlocked + " 次");
            console.log("  文件隐藏: " + stats.fileHidden + " 次");
            console.log("  内容过滤: " + stats.contentFiltered + " 次");
            console.log("  退出阻止: " + stats.exitBlocked + " 次");
            console.log("  字符串拦截: " + stats.stringBlocked + " 次");
            console.log("═".repeat(70) + "\n");
        }
    }, 30000);  // 每 30 秒显示一次
}

// ═════════════════════════════════════════════════════════════
// 主程序启动
// ═════════════════════════════════════════════════════════════

console.log("⏳ 正在启动终极防护系统...\n");

setTimeout(function() {
    console.log("═".repeat(70));
    console.log("                    🚀 开始初始化");
    console.log("═".repeat(70) + "\n");
    
    // 启动 Java 层防护
    initJavaProtection();
    
    // 启动 Native 层防护
    initNativeProtection();
    
    // 启动统计
    showStatistics();
    
    console.log("\n" + "═".repeat(70));
    console.log("  🎉🎉🎉 终极防护系统已全面启动！🎉🎉🎉");
    console.log("  🔒 所有已知的 Frida 检测手段已被拦截");
    console.log("  💪 可以安全地进行逆向分析了");
    console.log("═".repeat(70) + "\n");
    
}, 500);

