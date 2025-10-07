// 第十七关 - Patch get_number() 返回 42（简洁版）
// 适配 Frida 17.x

console.log("[★] 第十七关：Patch get_number() 返回 42\n");

var moduleName = "libfrida.so";
var base = Module.findBaseAddress(moduleName);

if (!base) {
    console.log("[-] 未找到 " + moduleName);
} else {
    console.log("[✓] " + moduleName + " 基地址: " + base + "\n");
    
    // 方法 1: 使用 Interceptor.replace() 替换整个函数
    console.log("[方法 1] 使用 Interceptor.replace()");
    console.log("=".repeat(60));
    
    var get_number = Module.findExportByName(moduleName, "_Z10get_numberv");
    
    if (!get_number) {
        // 尝试其他可能的符号名
        var exports = Module.enumerateExports(moduleName);
        exports.forEach(function(exp) {
            if (exp.name.indexOf("get_number") !== -1) {
                get_number = exp.address;
                console.log("找到: " + exp.name + " @ " + exp.address);
            }
        });
    }
    
    if (get_number) {
        console.log("get_number 地址: " + get_number + "\n");
        
        // 替换函数：直接返回 42
        Interceptor.replace(get_number, new NativeCallback(function() {
            console.log("[Patched] get_number() 被调用，返回 42");
            return 42;
        }, 'int', []));
        
        console.log("✓ 函数已 Patch！现在总是返回 42\n");
        
    } else {
        console.log("[-] 未找到 get_number 函数\n");
    }
}

// Hook Java 层验证
Java.perform(function() {
    var MainActivity = Java.use("cn.binary.frida.MainActivity");
    
    MainActivity.GetNumber.implementation = function() {
        console.log("\n[Java] MainActivity.GetNumber() 被调用");
        var result = this.GetNumber();
        console.log("[Java] 返回值: " + result);
        
        if (result === 42) {
            console.log("🎉 成功！返回值是 42！\n");
        } else {
            console.log("⚠️  返回值不是 42，Patch 可能失败\n");
        }
        
        return result;
    };
    
    console.log("[✓] Hook 完成，等待调用...\n");
});
