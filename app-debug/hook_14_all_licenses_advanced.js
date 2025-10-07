// 第十四关 - 获取所有许可证（高级版）
// 适配 Frida 17.x

console.log("[★] 第十四关：获取所有许可证列表 - 高级版");
console.log("=".repeat(70) + "\n");

var moduleName = "libfrida.so";
var base = Module.findBaseAddress(moduleName);

if (!base) {
    console.log("[-] 未找到模块");
} else {
    console.log("[✓] " + moduleName);
    console.log("  基地址: " + base + "\n");
    
    // ═════════════════════════════════════════════════════════════
    // 模块 1：枚举符号，查找内部函数
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 1] 枚举导出符号");
    console.log("─".repeat(70) + "\n");
    
    var exports = Module.enumerateExports(moduleName);
    
    var getLicenseList = null;
    var getLicense = null;
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf("getLicenseList") !== -1) {
            getLicenseList = exp.address;
            console.log("[找到] getLicenseList");
            console.log("  符号: " + exp.name);
            console.log("  地址: " + exp.address);
            console.log("");
        }
        
        // 尝试找到内部的 getLicense 函数
        if (exp.name.indexOf("getLicense") !== -1 && 
            exp.name.indexOf("List") === -1) {
            getLicense = exp.address;
            console.log("[找到] getLicense (内部函数)");
            console.log("  符号: " + exp.name);
            console.log("  地址: " + exp.address);
            console.log("");
        }
    });
    
    // ═════════════════════════════════════════════════════════════
    // 模块 2：Hook getLicenseList（查看过滤逻辑）
    // ═════════════════════════════════════════════════════════════
    
    if (getLicenseList) {
        console.log("[模块 2] Hook getLicenseList");
        console.log("─".repeat(70) + "\n");
        
        Interceptor.attach(getLicenseList, {
            onEnter: function(args) {
                console.log("\n[getLicenseList 被调用]");
                console.log("  JNIEnv: " + args[0]);
                console.log("  jobject: " + args[1]);
                console.log("");
            },
            
            onLeave: function(retval) {
                console.log("[getLicenseList 返回]");
                console.log("  jobjectArray: " + retval);
                
                // 读取数组
                Java.perform(function() {
                    try {
                        var env = Java.vm.getEnv();
                        var array = env.newLocalRef(retval);
                        var length = env.getArrayLength(array);
                        
                        console.log("\n  [返回的许可证列表]（已过滤）");
                        console.log("  ─".repeat(35));
                        console.log("  数量: " + length);
                        console.log("");
                        
                        for (var i = 0; i < length; i++) {
                            var element = env.getObjectArrayElement(array, i);
                            
                            if (element && !element.isNull()) {
                                var chars = env.getStringUtfChars(element, null);
                                var str = chars.readCString();
                                
                                console.log("    [" + i + "] " + str);
                                
                                if (str.indexOf("not allowed") !== -1) {
                                    console.log("        ⚠️  被过滤项");
                                } else if (str.indexOf("flag") !== -1) {
                                    console.log("        🚩 Flag!");
                                }
                                
                                env.releaseStringUtfChars(element, chars);
                                env.deleteLocalRef(element);
                            }
                        }
                        
                        console.log("  ─".repeat(35) + "\n");
                        env.deleteLocalRef(array);
                        
                    } catch (e) {
                        console.log("  [-] 读取失败: " + e);
                    }
                });
            }
        });
    }
    
    // ═════════════════════════════════════════════════════════════
    // 模块 3：主动调用 getLicense（绕过过滤）
    // ═════════════════════════════════════════════════════════════
    
    if (getLicense) {
        console.log("[模块 3] 使用 NativeFunction 主动调用 getLicense");
        console.log("─".repeat(70) + "\n");
        
        try {
            // 创建可调用函数
            // 签名: const char* getLicense(int index, const char* password)
            var getLicenseFunc = new NativeFunction(
                getLicense,
                'pointer',  // 返回 const char*
                ['int', 'pointer']  // (int index, const char* password)
            );
            
            console.log("[主动调用 getLicense]");
            console.log("═".repeat(70));
            
            var password = Memory.allocUtf8String("password");
            
            // 尝试获取所有许可证（0-9）
            console.log("\n获取所有原始许可证（未过滤）：\n");
            
            for (var i = 0; i < 10; i++) {
                try {
                    var resultPtr = getLicenseFunc(i, password);
                    
                    if (resultPtr && !resultPtr.isNull()) {
                        var license = resultPtr.readCString();
                        
                        console.log("[" + i + "] " + license);
                        
                        // 分析类型
                        if (license.indexOf("PRO") !== -1) {
                            console.log("     类型: PRO 版本 ⚠️  (会被过滤)");
                        } else if (license.indexOf("flag") !== -1) {
                            console.log("     类型: Flag 🚩");
                        } else {
                            console.log("     类型: 普通许可证");
                        }
                        
                        console.log("");
                    } else {
                        // 没有更多许可证了
                        break;
                    }
                } catch (e) {
                    console.log("[" + i + "] 调用失败: " + e.message);
                    break;
                }
            }
            
            console.log("═".repeat(70) + "\n");
            
        } catch (e) {
            console.log("[-] NativeFunction 调用失败: " + e);
            console.log(e.stack);
        }
    } else {
        console.log("[模块 3] 未找到 getLicense 内部函数");
        console.log("─".repeat(70));
        console.log("可能原因:");
        console.log("  • 函数未导出");
        console.log("  • 符号被剥离");
        console.log("  • 需要通过 IDA 分析获取偏移\n");
    }
    
    // ═════════════════════════════════════════════════════════════
    // 模块 4：对比分析
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 4] 对比分析");
    console.log("═".repeat(70));
    console.log("");
    console.log("getLicenseList() - 公开函数:");
    console.log("  • 会过滤掉包含 \"PRO\" 的许可证");
    console.log("  • 返回经过审查的列表");
    console.log("  • 用户只能看到部分内容");
    console.log("");
    console.log("getLicense() - 内部函数:");
    console.log("  • 直接返回原始许可证");
    console.log("  • 不进行任何过滤");
    console.log("  • 可以获取所有隐藏内容");
    console.log("");
    console.log("💡 绕过方法:");
    console.log("  1. 使用 NativeFunction 主动调用内部函数");
    console.log("  2. Hook getLicenseList 修改返回值");
    console.log("  3. 从 IDA 中找到偏移直接调用");
    console.log("═".repeat(70) + "\n");
    
    console.log("[✓] 所有模块已设置完成\n");
}


