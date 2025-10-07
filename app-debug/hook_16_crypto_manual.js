// 第十六关 - 手动指定偏移调用加密（如果符号被剥离）
// 适配 Frida 17.x

console.log("[★] 第十六关：手动指定偏移调用加密函数\n");

var moduleName = "libfrida.so";
var base = Module.findBaseAddress(moduleName);

if (!base) {
    console.log("[-] 未找到 " + moduleName);
} else {
    console.log("[模块基地址] " + base + "\n");
    
    // 方法 1: 如果知道偏移（从 IDA 中获取）
    console.log("[方法 1] 使用已知偏移");
    console.log("=".repeat(70));
    console.log("如果从 IDA 中看到函数偏移，可以这样调用：\n");
    
    // 示例偏移（需要根据实际 IDA 分析替换）
    var crypto_init_offset = 0x1000;   // 替换为实际偏移
    var crypto_crypt_offset = 0x1100;  // 替换为实际偏移
    
    var crypto_init_addr = base.add(crypto_init_offset);
    var crypto_crypt_addr = base.add(crypto_crypt_offset);
    
    console.log("crypto_init 地址: " + crypto_init_addr);
    console.log("crypto_crypt 地址: " + crypto_crypt_addr);
    console.log("");
    
    // 方法 2: Hook 现有的 JNI 函数来触发加密
    console.log("\n[方法 2] Hook JNI 函数来观察加密过程");
    console.log("=".repeat(70) + "\n");
    
    Java.perform(function() {
        // 查找调用加密的 Java 方法
        var MainActivity = Java.use("cn.binary.frida.MainActivity");
        
        // Hook GetNumber 方法（可能使用了加密）
        try {
            MainActivity.GetNumber.implementation = function() {
                console.log("\n[MainActivity.GetNumber 被调用]");
                var result = this.GetNumber();
                console.log("  返回值: " + result);
                return result;
            };
            console.log("✓ Hook MainActivity.GetNumber");
        } catch (e) {}
        
        // Hook processSensitiveData（可能使用了加密）
        try {
            MainActivity.processSensitiveData.implementation = function(str) {
                console.log("\n[MainActivity.processSensitiveData 被调用]");
                console.log("  输入: " + str);
                var result = this.processSensitiveData(str);
                console.log("  返回值: " + result);
                return result;
            };
            console.log("✓ Hook MainActivity.processSensitiveData");
        } catch (e) {}
        
        console.log("");
    });
    
    // 方法 3: 直接 Hook crypto_init 和 crypto_crypt
    console.log("\n[方法 3] 直接 Hook 加密函数");
    console.log("=".repeat(70) + "\n");
    
    var exports = Module.enumerateExports(moduleName);
    var hooked = false;
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf("crypto_init") !== -1) {
            console.log("[Hook] " + exp.name + " @ " + exp.address);
            
            Interceptor.attach(exp.address, {
                onEnter: function(args) {
                    console.log("\n[crypto_init 被调用]");
                    console.log("  参数 1 (result): " + args[0]);
                    console.log("  参数 2 (key): " + args[1]);
                    console.log("  参数 3 (key_len): " + args[2]);
                    
                    try {
                        var key = args[1].readCString();
                        console.log("  密钥内容: " + key);
                    } catch (e) {}
                    
                    this.result = args[0];
                },
                onLeave: function(retval) {
                    console.log("  返回值: " + retval);
                    
                    // 显示初始化后的结果
                    try {
                        console.log("\n  [初始化结果]");
                        console.log(hexdump(this.result, {
                            length: 64,
                            header: false,
                            ansi: true
                        }));
                    } catch (e) {}
                }
            });
            
            hooked = true;
        }
        
        if (exp.name.indexOf("crypto_crypt") !== -1) {
            console.log("[Hook] " + exp.name + " @ " + exp.address);
            
            Interceptor.attach(exp.address, {
                onEnter: function(args) {
                    console.log("\n[crypto_crypt 被调用]");
                    console.log("  参数 1 (result): " + args[0]);
                    console.log("  参数 2 (data): " + args[1]);
                    console.log("  参数 3 (data_len): " + args[2]);
                    
                    try {
                        var data = args[1].readCString();
                        console.log("  输入数据: " + data);
                    } catch (e) {
                        // 尝试读取字节
                        try {
                            console.log("  输入数据 (hex):");
                            console.log(hexdump(args[1], {
                                length: parseInt(args[2]),
                                header: false,
                                ansi: true
                            }));
                        } catch (e2) {}
                    }
                    
                    this.result = args[0];
                },
                onLeave: function(retval) {
                    console.log("  返回值: " + retval);
                    
                    // 显示加密结果
                    try {
                        console.log("\n  [加密结果]");
                        console.log(hexdump(this.result, {
                            length: 128,
                            header: false,
                            ansi: true
                        }));
                        
                        // 尝试读取字符串
                        var resultStr = this.result.readCString();
                        if (resultStr) {
                            console.log("\n  [加密结果字符串] " + resultStr);
                        }
                    } catch (e) {}
                }
            });
            
            hooked = true;
        }
    });
    
    if (hooked) {
        console.log("\n[✓] Hook 设置完成，等待函数调用...\n");
    } else {
        console.log("\n[-] 未找到加密函数符号\n");
        console.log("请检查以下可能的函数：\n");
        
        exports.forEach(function(exp, i) {
            if (i < 30) {
                console.log("  [" + i + "] " + exp.name);
            }
        });
        console.log("");
    }
}

