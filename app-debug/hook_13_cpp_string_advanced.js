// 第十三关 - Hook C++ std::string（高级版）
// 适配 Frida 17.x

console.log("[★] 第十三关：Hook C++ std::string - 高级版");
console.log("=".repeat(70) + "\n");

var moduleName = "libfrida.so";
var base = Module.findBaseAddress(moduleName);

if (!base) {
    console.log("[-] 未找到模块");
} else {
    console.log("[✓] " + moduleName + " 基地址: " + base + "\n");
    
    // ═════════════════════════════════════════════════════════════
    // 模块 1：查找并Hook所有JNI函数
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 1] 枚举所有 JNI 函数");
    console.log("─".repeat(70) + "\n");
    
    var exports = Module.enumerateExports(moduleName);
    var jniFunctions = [];
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf("Java_cn_binary_frida") !== -1) {
            jniFunctions.push(exp);
            console.log("[找到] " + exp.name);
            console.log("  地址: " + exp.address);
            console.log("  偏移: 0x" + exp.address.sub(base).toString(16));
            console.log("");
        }
    });
    
    console.log("共找到 " + jniFunctions.length + " 个 JNI 函数\n");
    
    // ═════════════════════════════════════════════════════════════
    // 模块 2：Hook stringFromJNI
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 2] Hook stringFromJNI");
    console.log("─".repeat(70) + "\n");
    
    jniFunctions.forEach(function(exp) {
        if (exp.name.indexOf("stringFromJNI") !== -1) {
            console.log("[Hook] " + exp.name);
            
            Interceptor.attach(exp.address, {
                onEnter: function(args) {
                    console.log("\n[stringFromJNI 被调用]");
                    console.log("  JNIEnv: " + args[0]);
                    console.log("  jobject: " + args[1]);
                },
                
                onLeave: function(retval) {
                    console.log("  返回值(jstring): " + retval);
                    
                    // 使用 JNI 读取字符串
                    Java.perform(function() {
                        try {
                            var env = Java.vm.getEnv();
                            var jstr = env.newLocalRef(retval);
                            var chars = env.getStringUtfChars(jstr, null);
                            var result = chars.readCString();
                            
                            console.log("  🔑 字符串内容: \"" + result + "\"");
                            
                            env.releaseStringUtfChars(jstr, chars);
                            env.deleteLocalRef(jstr);
                        } catch (e) {
                            console.log("  [-] 读取失败: " + e);
                        }
                    });
                    
                    console.log("");
                }
            });
        }
    });
    
    // ═════════════════════════════════════════════════════════════
    // 模块 3：Hook getLicenseList
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 3] Hook getLicenseList");
    console.log("─".repeat(70) + "\n");
    
    jniFunctions.forEach(function(exp) {
        if (exp.name.indexOf("getLicenseList") !== -1) {
            console.log("[Hook] " + exp.name);
            
            Interceptor.attach(exp.address, {
                onEnter: function(args) {
                    console.log("\n[getLicenseList 被调用]");
                },
                
                onLeave: function(retval) {
                    console.log("  返回值(jobjectArray): " + retval);
                    
                    // 读取数组内容
                    Java.perform(function() {
                        try {
                            var env = Java.vm.getEnv();
                            var array = env.newLocalRef(retval);
                            var length = env.getArrayLength(array);
                            
                            console.log("  数组长度: " + length);
                            console.log("  ─".repeat(35));
                            
                            for (var i = 0; i < length; i++) {
                                var element = env.getObjectArrayElement(array, i);
                                
                                if (element && !element.isNull()) {
                                    var chars = env.getStringUtfChars(element, null);
                                    var str = chars.readCString();
                                    
                                    console.log("    [" + i + "] " + str);
                                    
                                    if (str.indexOf("flag") !== -1) {
                                        console.log("        🚩 包含 flag!");
                                    }
                                    
                                    env.releaseStringUtfChars(element, chars);
                                    env.deleteLocalRef(element);
                                }
                            }
                            
                            console.log("  ─".repeat(35));
                            env.deleteLocalRef(array);
                        } catch (e) {
                            console.log("  [-] 读取失败: " + e);
                        }
                    });
                    
                    console.log("");
                }
            });
        }
    });
    
    // ═════════════════════════════════════════════════════════════
    // 模块 4：Hook processSensitiveData
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 4] Hook processSensitiveData");
    console.log("─".repeat(70) + "\n");
    
    jniFunctions.forEach(function(exp) {
        if (exp.name.indexOf("processSensitiveData") !== -1) {
            console.log("[Hook] " + exp.name);
            
            Interceptor.attach(exp.address, {
                onEnter: function(args) {
                    console.log("\n[processSensitiveData 被调用]");
                    
                    // 读取输入参数
                    Java.perform(function() {
                        try {
                            var env = Java.vm.getEnv();
                            var jstr = env.newLocalRef(args[2]);
                            var chars = env.getStringUtfChars(jstr, null);
                            var input = chars.readCString();
                            
                            console.log("  输入: \"" + input + "\"");
                            
                            env.releaseStringUtfChars(jstr, chars);
                            env.deleteLocalRef(jstr);
                        } catch (e) {}
                    });
                },
                
                onLeave: function(retval) {
                    // 读取返回值
                    Java.perform(function() {
                        try {
                            var env = Java.vm.getEnv();
                            var jstr = env.newLocalRef(retval);
                            var chars = env.getStringUtfChars(jstr, null);
                            var output = chars.readCString();
                            
                            console.log("  输出: \"" + output + "\"");
                            
                            if (output.indexOf("flag{") !== -1) {
                                console.log("  🚩🚩🚩 发现 FLAG!");
                            }
                            
                            env.releaseStringUtfChars(jstr, chars);
                            env.deleteLocalRef(jstr);
                        } catch (e) {}
                    });
                    
                    console.log("");
                }
            });
        }
    });
    
    console.log("[✓] 所有 Hook 已设置\n");
    
    // ═════════════════════════════════════════════════════════════
    // 说明
    // ═════════════════════════════════════════════════════════════
    
    console.log("═".repeat(70));
    console.log("[JNI 字符串处理技巧]");
    console.log("═".repeat(70));
    console.log("• jstring → C string:");
    console.log("    env.getStringUtfChars(jstr, null)");
    console.log("");
    console.log("• jobjectArray → 遍历:");
    console.log("    env.getArrayLength(array)");
    console.log("    env.getObjectArrayElement(array, i)");
    console.log("");
    console.log("• 记得释放资源:");
    console.log("    env.releaseStringUtfChars(jstr, chars)");
    console.log("    env.deleteLocalRef(jstr)");
    console.log("═".repeat(70) + "\n");
}

