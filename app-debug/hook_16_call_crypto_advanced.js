// 第十六关 - 主动调用加密函数（高级版）
// 适配 Frida 17.x

console.log("[★] 第十六关：主动调用 Native 加密函数 - 高级版");
console.log("=".repeat(70) + "\n");

var moduleName = "libfrida.so";
var base = Module.findBaseAddress(moduleName);

if (!base) {
    console.log("[-] 未找到模块");
} else {
    console.log("[✓] " + moduleName);
    console.log("  基地址: " + base + "\n");
    
    // ═════════════════════════════════════════════════════════════
    // 模块 1：查找加密相关函数
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 1] 枚举加密相关函数");
    console.log("─".repeat(70) + "\n");
    
    var exports = Module.enumerateExports(moduleName);
    
    var crypto_init = null;
    var crypto_crypt = null;
    var allCryptoFuncs = [];
    
    exports.forEach(function(exp) {
        if (exp.name.toLowerCase().indexOf("crypto") !== -1 || 
            exp.name.toLowerCase().indexOf("encrypt") !== -1 ||
            exp.name.toLowerCase().indexOf("crypt") !== -1) {
            
            allCryptoFuncs.push(exp);
            
            console.log("[找到] " + exp.name);
            console.log("  地址: " + exp.address);
            console.log("  偏移: 0x" + exp.address.sub(base).toString(16));
            
            if (exp.name.indexOf("crypto_init") !== -1) {
                crypto_init = exp.address;
                console.log("  → crypto_init ✓");
            }
            
            if (exp.name.indexOf("crypto_crypt") !== -1) {
                crypto_crypt = exp.address;
                console.log("  → crypto_crypt ✓");
            }
            
            console.log("");
        }
    });
    
    console.log("共找到 " + allCryptoFuncs.length + " 个加密相关函数\n");
    
    // ═════════════════════════════════════════════════════════════
    // 模块 2：Hook 加密函数观察行为
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 2] Hook 加密函数");
    console.log("─".repeat(70) + "\n");
    
    if (crypto_init) {
        Interceptor.attach(crypto_init, {
            onEnter: function(args) {
                console.log("\n[crypto_init Hook]");
                console.log("  result: " + args[0]);
                console.log("  key: " + args[1]);
                console.log("  key_len: " + args[2]);
                
                try {
                    var keyLen = parseInt(args[2]);
                    if (keyLen > 0 && keyLen < 64) {
                        var key = args[1].readByteArray(keyLen);
                        console.log("  密钥内容:");
                        console.log(hexdump(key, {
                            length: keyLen,
                            header: false,
                            ansi: true
                        }));
                    }
                } catch (e) {}
            }
        });
    }
    
    if (crypto_crypt) {
        Interceptor.attach(crypto_crypt, {
            onEnter: function(args) {
                this.result = args[0];
                this.data = args[1];
                this.len = parseInt(args[2]);
                
                console.log("\n[crypto_crypt Hook]");
                console.log("  result: " + this.result);
                console.log("  data: " + this.data);
                console.log("  data_len: " + this.len);
                
                if (this.len > 0 && this.len < 256) {
                    console.log("\n  输入数据:");
                    console.log(hexdump(this.data, {
                        length: this.len,
                        header: false,
                        ansi: true
                    }));
                }
            },
            
            onLeave: function(retval) {
                console.log("\n  返回值: " + retval);
                
                if (this.result && this.len > 0 && this.len < 256) {
                    console.log("\n  输出数据:");
                    console.log(hexdump(this.result, {
                        length: this.len,
                        header: false,
                        ansi: true
                    }));
                }
                
                console.log("\n" + "─".repeat(70));
            }
        });
    }
    
    // ═════════════════════════════════════════════════════════════
    // 模块 3：主动调用加密函数
    // ═════════════════════════════════════════════════════════════
    
    if (crypto_init && crypto_crypt) {
        console.log("[模块 3] 主动调用加密函数");
        console.log("═".repeat(70) + "\n");
        
        setTimeout(function() {
            try {
                // 创建函数
                var initFunc = new NativeFunction(crypto_init, 'int64', ['pointer', 'pointer', 'uint64']);
                var cryptFunc = new NativeFunction(crypto_crypt, 'int64', ['pointer', 'pointer', 'uint64']);
                
                // 测试用例
                var testCases = [
                    { data: "Hello World", key: "key123" },
                    { data: "Frida Hook", key: "secret" },
                    { data: "flag{test}", key: "password" },
                    { data: "12345678", key: "mykey" }
                ];
                
                testCases.forEach(function(test, idx) {
                    console.log("[测试 #" + (idx + 1) + "]");
                    console.log("─".repeat(70));
                    console.log("输入: \"" + test.data + "\"");
                    console.log("密钥: \"" + test.key + "\"");
                    console.log("");
                    
                    // 分配内存
                    var resultBuffer = Memory.alloc(512);
                    var dataBuffer = Memory.allocUtf8String(test.data);
                    var keyBuffer = Memory.allocUtf8String(test.key);
                    
                    // 初始化
                    console.log("[调用 crypto_init]");
                    var initResult = initFunc(resultBuffer, keyBuffer, test.key.length);
                    console.log("  返回值: " + initResult);
                    
                    // 加密
                    console.log("\n[调用 crypto_crypt]");
                    var cryptResult = cryptFunc(resultBuffer, dataBuffer, test.data.length);
                    console.log("  返回值: " + cryptResult);
                    
                    // 显示结果
                    console.log("\n[加密结果]");
                    console.log(hexdump(resultBuffer, {
                        length: Math.min(64, test.data.length * 2),
                        header: true,
                        ansi: true
                    }));
                    
                    // 分析
                    console.log("\n[分析]");
                    var encrypted = resultBuffer.readByteArray(test.data.length);
                    var original = dataBuffer.readByteArray(test.data.length);
                    
                    console.log("  原始数据: " + Array.from(new Uint8Array(original)).map(b => 
                        ("0" + b.toString(16)).slice(-2)).join(" "));
                    console.log("  加密数据: " + Array.from(new Uint8Array(encrypted)).map(b => 
                        ("0" + b.toString(16)).slice(-2)).join(" "));
                    
                    // XOR 检测
                    var xorValues = [];
                    var origArray = new Uint8Array(original);
                    var encArray = new Uint8Array(encrypted);
                    
                    for (var i = 0; i < Math.min(origArray.length, encArray.length); i++) {
                        xorValues.push(origArray[i] ^ encArray[i]);
                    }
                    
                    console.log("  XOR 值: " + xorValues.map(v => 
                        ("0" + v.toString(16)).slice(-2)).join(" "));
                    
                    // 判断是否是简单 XOR
                    var allSame = xorValues.every(function(v) { return v === xorValues[0]; });
                    if (allSame && xorValues.length > 0) {
                        console.log("  🔍 检测到固定 XOR: 0x" + xorValues[0].toString(16));
                    }
                    
                    console.log("\n" + "═".repeat(70) + "\n");
                });
                
                console.log("[✓] 主动调用测试完成\n");
                
            } catch (e) {
                console.log("[-] 主动调用失败: " + e);
                console.log(e.stack);
            }
        }, 1000);
    } else {
        console.log("[模块 3] 未找到完整的加密函数");
        console.log("  crypto_init: " + (crypto_init ? "✓" : "✗"));
        console.log("  crypto_crypt: " + (crypto_crypt ? "✓" : "✗"));
        console.log("");
    }
    
    // ═════════════════════════════════════════════════════════════
    // 总结
    // ═════════════════════════════════════════════════════════════
    
    console.log("[技术总结]");
    console.log("═".repeat(70));
    console.log("• NativeFunction: 创建可调用的 Native 函数");
    console.log("• Memory.alloc(): 分配内存缓冲区");
    console.log("• Memory.allocUtf8String(): 分配字符串");
    console.log("• hexdump(): 查看内存内容");
    console.log("• readByteArray(): 读取字节数据");
    console.log("═".repeat(70) + "\n");
    
    console.log("[✓] 所有模块已设置\n");
}


