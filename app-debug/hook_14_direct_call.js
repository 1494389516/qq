// 第十四关 - 主动调用获取所有许可证

console.log("[★] 第十四关：主动调用获取许可证");
console.log("=".repeat(60) + "\n");

var moduleName = "libfrida.so";

// 查找 getLicense 函数
var exports = Module.enumerateExports(moduleName);
var getLicenseAddr = null;

exports.forEach(function(exp) {
    if (exp.name.indexOf("getLicense") !== -1 && 
        exp.name.indexOf("List") === -1) {
        getLicenseAddr = exp.address;
        console.log("找到 getLicense: " + exp.address);
        console.log("符号: " + exp.name + "\n");
    }
});

if (getLicenseAddr) {
    console.log("[主动调用 getLicense]");
    console.log("=".repeat(60));
    
    // 创建 NativeFunction
    var getLicense = new NativeFunction(getLicenseAddr, 'pointer', ['int', 'pointer']);
    
    // 准备参数
    var password = Memory.allocUtf8String("password");
    
    console.log("\n获取所有许可证：\n");
    
    // 调用 3 次获取所有许可证
    for (var i = 0; i < 3; i++) {
        try {
            var resultPtr = getLicense(i, password);
            var license = resultPtr.readCString();
            
            console.log("[" + i + "] " + license);
            
            // 分析许可证内容
            if (license.indexOf("PRO") !== -1) {
                console.log("     类型: PRO 版本 ⚠️  (会被过滤)");
            } else if (license.indexOf("flag") !== -1) {
                console.log("     类型: Flag 🚩");
            } else {
                console.log("     类型: 普通许可证");
            }
            console.log("");
            
        } catch (e) {
            console.log("[" + i + "] 调用失败: " + e);
        }
    }
    
    console.log("=".repeat(60));
    
} else {
    console.log("✗ 未找到 getLicense 函数");
}

// 同时 Hook 查看正常调用
console.log("\n[设置 Hook 监控]");

var jniFunc = Module.findExportByName(moduleName,
    "Java_cn_binary_frida_MainActivity_getLicenseList");

if (jniFunc) {
    Interceptor.attach(jniFunc, {
        onLeave: function(retval) {
            console.log("\n[getLicenseList 被调用 - 返回的过滤后数组]");
            
            Java.perform(function() {
                var env = Java.vm.getEnv();
                var arr = env.newLocalRef(retval);
                var len = env.getArrayLength(arr);
                
                for (var i = 0; i < len; i++) {
                    var element = env.getObjectArrayElement(arr, i);
                    var chars = env.getStringUtfChars(element, null);
                    var str = chars.readCString();
                    
                    console.log("  [" + i + "] " + str);
                    env.releaseStringUtfChars(element, chars);
                }
                console.log("");
            });
        }
    });
}

console.log("\n[✓] 完成\n");
