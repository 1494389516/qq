// 🛡️ Frida 反检测 - 简洁版
// 适配 Frida 17.x
// 快速部署，对抗最常见的检测

console.log("[🛡️] Frida 反检测系统 - 简洁版\n");

Java.perform(function() {
    
    // ═══════════════════════════════════════
    // 1. 阻止端口检测（最常见）
    // ═══════════════════════════════════════
    
    console.log("[1/5] 端口检测防护...");
    try {
        var Socket = Java.use("java.net.Socket");
        Socket.$init.overload('java.lang.String', 'int').implementation = function(host, port) {
            if (port === 27042 || port === 27043) {
                console.log("  ✓ 拦截 Frida 端口: " + port);
                throw Java.use("java.net.ConnectException").$new("Connection refused");
            }
            return this.$init(host, port);
        };
        console.log("  ✅ 完成\n");
    } catch (e) {
        console.log("  ❌ 失败\n");
    }
    
    // ═══════════════════════════════════════
    // 2. 隐藏 Frida 文件
    // ═══════════════════════════════════════
    
    console.log("[2/5] 文件检测防护...");
    try {
        var File = Java.use("java.io.File");
        File.exists.implementation = function() {
            var path = this.getAbsolutePath().toString();
            if (path.toLowerCase().includes("frida")) {
                console.log("  ✓ 隐藏文件: " + path);
                return false;
            }
            return this.exists();
        };
        console.log("  ✅ 完成\n");
    } catch (e) {
        console.log("  ❌ 失败\n");
    }
    
    // ═══════════════════════════════════════
    // 3. 过滤文件内容
    // ═══════════════════════════════════════
    
    console.log("[3/5] 内容过滤防护...");
    try {
        var BufferedReader = Java.use("java.io.BufferedReader");
        BufferedReader.readLine.overload().implementation = function() {
            var line = this.readLine();
            if (line && line.toLowerCase().includes("frida")) {
                return line.replace(/frida/gi, "xxxxx");
            }
            return line;
        };
        console.log("  ✅ 完成\n");
    } catch (e) {
        console.log("  ❌ 失败\n");
    }
    
    // ═══════════════════════════════════════
    // 4. 阻止应用退出
    // ═══════════════════════════════════════
    
    console.log("[4/5] 退出保护...");
    try {
        var System = Java.use("java.lang.System");
        System.exit.implementation = function(code) {
            console.log("  ✓ 阻止 System.exit(" + code + ")");
            // 不执行退出
        };
        
        var Process = Java.use("android.os.Process");
        Process.killProcess.implementation = function(pid) {
            console.log("  ✓ 阻止 killProcess(" + pid + ")");
            // 不执行终止
        };
        console.log("  ✅ 完成\n");
    } catch (e) {
        console.log("  ❌ 失败\n");
    }
    
    // ═══════════════════════════════════════
    // 5. Hook 字符串比较（Native 层）
    // ═══════════════════════════════════════
    
    console.log("[5/5] 字符串检测防护...");
    try {
        var strstr = Module.findExportByName("libc.so", "strstr");
        if (strstr) {
            Interceptor.attach(strstr, {
                onEnter: function(args) {
                    var needle = args[1].readCString();
                    if (needle && needle.toLowerCase().includes("frida")) {
                        this.block = true;
                    }
                },
                onLeave: function(retval) {
                    if (this.block) {
                        retval.replace(NULL);
                    }
                }
            });
        }
        console.log("  ✅ 完成\n");
    } catch (e) {
        console.log("  ❌ 失败\n");
    }
    
    console.log("═".repeat(50));
    console.log("🎉 反检测系统已启动！");
    console.log("═".repeat(50) + "\n");
});

