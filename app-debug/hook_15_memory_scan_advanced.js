// 第十五关 - 高级内存扫描技术
// 适配 Frida 17.x

console.log("[★] 第十五关：高级内存扫描\n");

Java.perform(function() {
    
    // ===============================================
    // 技巧 1: 搜索特定类的实例并读取字段
    // ===============================================
    
    console.log("[技巧 1] 扫描 Java 对象实例");
    console.log("=".repeat(70));
    
    setTimeout(function() {
        try {
            // 搜索可能包含 flag 的类
            var targetClasses = [
                "cn.binary.frida.SensitiveDataProcessor",
                "cn.binary.frida.Login",
                "cn.binary.frida.Utils",
                "cn.binary.frida.MainActivity"
            ];
            
            targetClasses.forEach(function(className) {
                try {
                    var instances = Java.choose(className);
                    
                    if (instances.length > 0) {
                        console.log("\n[" + className + "] 找到 " + instances.length + " 个实例");
                        
                        instances.forEach(function(instance, idx) {
                            console.log("\n  实例 #" + (idx + 1) + ":");
                            
                            // 获取类定义
                            var clazz = Java.use(className);
                            
                            // 尝试读取常见的字段名
                            var fieldNames = [
                                "flag", "FLAG", "secret", "SECRET", 
                                "key", "KEY", "password", "token",
                                "data", "result", "value"
                            ];
                            
                            fieldNames.forEach(function(fieldName) {
                                try {
                                    var fieldValue = instance[fieldName].value;
                                    if (fieldValue) {
                                        console.log("    " + fieldName + ": " + fieldValue);
                                        
                                        if (fieldValue.toString().indexOf("flag{") !== -1) {
                                            console.log("    🚩 发现 FLAG!");
                                        }
                                    }
                                } catch (e) {}
                            });
                            
                            // 尝试调用常见方法
                            var methodNames = ["toString", "getData", "getValue"];
                            methodNames.forEach(function(methodName) {
                                try {
                                    var result = instance[methodName]();
                                    if (result && result.toString().indexOf("flag") !== -1) {
                                        console.log("    " + methodName + "(): " + result);
                                    }
                                } catch (e) {}
                            });
                        });
                    }
                } catch (e) {}
            });
            
        } catch (e) {
            console.log("[-] 对象扫描失败: " + e);
        }
    }, 1000);
    
    
    // ===============================================
    // 技巧 2: 监控字符串分配（找新创建的 flag）
    // ===============================================
    
    console.log("\n\n[技巧 2] 监控字符串分配");
    console.log("=".repeat(70));
    console.log("Hook String 构造函数，捕获包含 'flag' 的字符串\n");
    
    var String = Java.use("java.lang.String");
    
    // Hook String 构造函数
    String.$init.overload('[B').implementation = function(bytes) {
        var result = this.$init(bytes);
        var str = this.toString();
        
        if (str && str.length > 0 && str.length < 200) {
            if (str.toLowerCase().indexOf("flag") !== -1 || 
                str.toLowerCase().indexOf("key") !== -1 ||
                str.toLowerCase().indexOf("secret") !== -1) {
                
                console.log("[String 创建] " + str);
                
                if (str.match(/flag\{[^}]+\}/i)) {
                    console.log("🚩🚩🚩 FLAG: " + str);
                }
            }
        }
        
        return result;
    };
    
    
    // ===============================================
    // 技巧 3: 搜索正则表达式模式
    // ===============================================
    
    console.log("\n[技巧 3] 使用正则表达式搜索");
    console.log("=".repeat(70));
    
    // 定义要搜索的正则模式
    var patterns = [
        /flag\{[a-zA-Z0-9_]+\}/,
        /FLAG\{[a-zA-Z0-9_]+\}/,
        /[a-f0-9]{32}/,  // MD5 格式
        /[a-f0-9]{64}/   // SHA256 格式
    ];
    
    // 搜索所有可读内存区域
    var readableRanges = Process.enumerateRanges('r--');
    var foundCount = 0;
    
    console.log("正在搜索 " + readableRanges.length + " 个内存区域...\n");
    
    readableRanges.slice(0, 20).forEach(function(range) {
        try {
            // 读取内存
            var data = range.base.readCString(Math.min(range.size, 4096));
            
            if (data) {
                patterns.forEach(function(pattern, idx) {
                    var matches = data.match(pattern);
                    if (matches) {
                        foundCount++;
                        console.log("[匹配 #" + foundCount + "] 模式 " + idx);
                        console.log("  地址: " + range.base);
                        console.log("  内容: " + matches[0]);
                        console.log("");
                    }
                });
            }
        } catch (e) {}
    });
    
    
    // ===============================================
    // 技巧 4: 搜索特定模块的数据段
    // ===============================================
    
    console.log("\n[技巧 4] 搜索模块数据段");
    console.log("=".repeat(70));
    
    var libfrida = Process.findModuleByName("libfrida.so");
    
    if (libfrida) {
        console.log("libfrida.so 找到");
        console.log("  基地址: " + libfrida.base);
        console.log("  大小: " + libfrida.size + " 字节\n");
        
        // 枚举 section
        var sections = libfrida.enumerateRanges('r--');
        console.log("找到 " + sections.length + " 个可读段\n");
        
        sections.forEach(function(section, idx) {
            // 搜索 "flag" 字符串
            try {
                Memory.scan(section.base, section.size, "66 6c 61 67", {
                    onMatch: function(address, size) {
                        console.log("[段 #" + idx + "] 找到匹配");
                        console.log("  地址: " + address);
                        console.log("  偏移: 0x" + address.sub(libfrida.base).toString(16));
                        
                        try {
                            var str = address.readCString(100);
                            console.log("  内容: " + str + "\n");
                        } catch (e) {}
                    },
                    onComplete: function() {}
                });
            } catch (e) {}
        });
    }
    
    
    console.log("\n" + "=".repeat(70));
    console.log("[✓] 高级内存扫描完成！");
    console.log("=".repeat(70) + "\n");
});

