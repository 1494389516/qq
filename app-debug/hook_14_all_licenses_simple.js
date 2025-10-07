// 第十四关 - 打印所有许可证（简洁版）

console.log("[★] 第十四关：获取完整许可证列表\n");

var moduleName = "libfrida.so";
var licenses = [];  // 存储原始许可证

// 1. Hook getLicense 函数（获取原始数据）
var exports = Module.enumerateExports(moduleName);
var getLicenseAddr = null;

exports.forEach(function(exp) {
    if (exp.name.indexOf("getLicense") !== -1 && 
        exp.name.indexOf("List") === -1) {
        getLicenseAddr = exp.address;
        console.log("找到 getLicense: " + exp.address + "\n");
    }
});

if (getLicenseAddr) {
    Interceptor.attach(getLicenseAddr, {
        onEnter: function(args) {
            this.index = args[0].toInt32();
            this.password = args[1].readCString();
        },
        onLeave: function(retval) {
            var license = retval.readCString();
            licenses[this.index] = license;
            
            console.log("[" + this.index + "] " + license);
            if (license.indexOf("PRO") !== -1) {
                console.log("     ⚠️  包含 PRO，会被过滤\n");
            }
        }
    });
}

// 2. Hook JNI getLicenseList（查看最终结果）
var jniFunc = Module.findExportByName(moduleName,
    "Java_cn_binary_frida_MainActivity_getLicenseList");

if (jniFunc) {
    Interceptor.attach(jniFunc, {
        onLeave: function(retval) {
            console.log("\n" + "=".repeat(50));
            console.log("【原始许可证 vs 返回结果】");
            console.log("=".repeat(50));
            
            Java.perform(function() {
                var env = Java.vm.getEnv();
                var arr = env.newLocalRef(retval);
                var len = env.getArrayLength(arr);
                
                for (var i = 0; i < len; i++) {
                    var element = env.getObjectArrayElement(arr, i);
                    var chars = env.getStringUtfChars(element, null);
                    var returned = chars.readCString();
                    
                    console.log("\n[" + i + "]");
                    console.log("  原始: " + licenses[i]);
                    console.log("  返回: " + returned);
                    
                    if (returned === "not allowed") {
                        console.log("  状态: ❌ 被过滤");
                    } else {
                        console.log("  状态: ✓ 正常");
                    }
                    
                    env.releaseStringUtfChars(element, chars);
                }
                
                console.log("\n" + "=".repeat(50));
            });
        }
    });
}

console.log("[✓] Hook 完成\n");
