// 第十三关 - Hook Native JNI 函数获取许可证信息（完整版）

console.log("[★] 第十三关：Hook JNI Native 函数");
console.log("=".repeat(60) + "\n");

var moduleName = "libfrida.so";
var packageName = "cn_binary_frida";

// JNI 函数名规则：Java_包名_类名_方法名
var jniFunctions = [
    "Java_cn_binary_frida_MainActivity_stringFromJNI",
    "Java_cn_binary_frida_MainActivity_getLicenseList",
    "Java_cn_binary_frida_MainActivity_processSensitiveData",
    "Java_cn_binary_frida_MainActivity_GetNumber"
];

console.log("[查找 JNI 函数]");
console.log("-".repeat(60));

var foundCount = 0;

jniFunctions.forEach(function(funcName) {
    var addr = Module.findExportByName(moduleName, funcName);
    
    if (addr) {
        foundCount++;
        console.log("✓ " + funcName);
        console.log("  地址: " + addr);
        console.log("");
        
        // Hook stringFromJNI
        if (funcName.indexOf("stringFromJNI") !== -1) {
            Interceptor.attach(addr, {
                onEnter: function(args) {
                    console.log("\n" + "=".repeat(50));
                    console.log("[stringFromJNI 被调用]");
                    console.log("=".repeat(50));
                },
                onLeave: function(retval) {
                    // retval 是 jstring
                    console.log("  返回值 (jstring): " + retval);
                    
                    // 通过 JNI 读取字符串内容
                    Java.perform(function() {
                        var env = Java.vm.getEnv();
                        try {
                            var StringClass = env.findClass("java/lang/String");
                            var jstr = env.newLocalRef(retval);
                            
                            // 调用 JNI 函数获取字符串
                            var chars = env.getStringUtfChars(jstr, null);
                            var result = chars.readCString();
                            
                            console.log("\n  🔑 许可证字符串: " + result);
                            console.log("=".repeat(50) + "\n");
                            
                            env.releaseStringUtfChars(jstr, chars);
                        } catch (e) {
                            console.log("  读取失败: " + e);
                        }
                    });
                }
            });
        }
        
        // Hook getLicenseList
        if (funcName.indexOf("getLicenseList") !== -1) {
            Interceptor.attach(addr, {
                onEnter: function(args) {
                    console.log("\n" + "=".repeat(50));
                    console.log("[getLicenseList 被调用]");
                    console.log("=".repeat(50));
                },
                onLeave: function(retval) {
                    console.log("  返回值 (jobjectArray): " + retval);
                    
                    Java.perform(function() {
                        var env = Java.vm.getEnv();
                        try {
                            var arr = env.newLocalRef(retval);
                            var len = env.getArrayLength(arr);
                            
                            console.log("  🔑 许可证列表 (" + len + " 项):");
                            
                            for (var i = 0; i < len; i++) {
                                var element = env.getObjectArrayElement(arr, i);
                                var chars = env.getStringUtfChars(element, null);
                                var str = chars.readCString();
                                console.log("    [" + i + "] " + str);
                                env.releaseStringUtfChars(element, chars);
                            }
                            
                            console.log("=".repeat(50) + "\n");
                        } catch (e) {
                            console.log("  读取失败: " + e);
                        }
                    });
                }
            });
        }
        
        // Hook processSensitiveData
        if (funcName.indexOf("processSensitiveData") !== -1) {
            Interceptor.attach(addr, {
                onEnter: function(args) {
                    // args[1] = jobject (this)
                    // args[2] = jstring (参数)
                    
                    console.log("\n" + "=".repeat(50));
                    console.log("[processSensitiveData 被调用]");
                    
                    Java.perform(function() {
                        var env = Java.vm.getEnv();
                        try {
                            var jstr = env.newLocalRef(args[2]);
                            var chars = env.getStringUtfChars(jstr, null);
                            var input = chars.readCString();
                            console.log("  输入参数: " + input);
                            env.releaseStringUtfChars(jstr, chars);
                        } catch (e) {}
                    });
                    
                    console.log("=".repeat(50));
                },
                onLeave: function(retval) {
                    Java.perform(function() {
                        var env = Java.vm.getEnv();
                        try {
                            var jstr = env.newLocalRef(retval);
                            var chars = env.getStringUtfChars(jstr, null);
                            var result = chars.readCString();
                            
                            console.log("\n  🔑 敏感数据: " + result);
                            console.log("=".repeat(50) + "\n");
                            
                            env.releaseStringUtfChars(jstr, chars);
                        } catch (e) {
                            console.log("  读取失败: " + e);
                        }
                    });
                }
            });
        }
        
        // Hook GetNumber
        if (funcName.indexOf("GetNumber") !== -1) {
            Interceptor.attach(addr, {
                onEnter: function(args) {
                    console.log("\n[GetNumber 被调用]");
                },
                onLeave: function(retval) {
                    console.log("  返回值: " + retval.toInt32() + "\n");
                }
            });
        }
        
    } else {
        console.log("✗ " + funcName + " (未找到)");
    }
});

console.log("");
console.log("=".repeat(60));
console.log("[统计] 找到 " + foundCount + " / " + jniFunctions.length + " 个 JNI 函数");
console.log("=".repeat(60));
console.log("\n[✓] Hook 完成，等待调用...\n");
