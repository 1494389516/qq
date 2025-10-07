// 第十七关 - 内存 Patch 汇编指令
// 适配 Frida 17.x

console.log("[★] 第十七关：内存 Patch - 修改汇编指令");
console.log("=".repeat(70) + "\n");

var moduleName = "libfrida.so";
var base = Module.findBaseAddress(moduleName);

if (!base) {
    console.log("[-] 未找到 " + moduleName);
} else {
    // 查找函数
    var get_number = null;
    var exports = Module.enumerateExports(moduleName);
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf("get_number") !== -1) {
            get_number = exp.address;
            console.log("[找到] " + exp.name + " @ " + exp.address + "\n");
        }
    });
    
    if (!get_number) {
        console.log("[-] 未找到 get_number");
        return;
    }
    
    // 显示原始汇编
    console.log("[原始汇编]");
    console.log("=".repeat(70));
    console.log(hexdump(get_number, {
        length: 96,
        header: true,
        ansi: true
    }));
    console.log("");
    
    // 检测架构
    var arch = Process.arch;
    console.log("[系统架构] " + arch + "\n");
    
    if (arch === 'arm64') {
        console.log("[Patch] 使用 ARM64 汇编");
        console.log("=".repeat(70));
        console.log("将函数开头替换为：");
        console.log("  mov w0, #42");
        console.log("  ret");
        console.log("");
        
        try {
            Memory.patchCode(get_number, 16, function(code) {
                var writer = new Arm64Writer(code, { pc: get_number });
                
                // mov w0, #42  (将 42 加载到返回寄存器)
                writer.putMovRegU32('w0', 42);
                
                // ret  (返回)
                writer.putRet();
                
                writer.flush();
                
                console.log("✓ Patch 成功！\n");
            });
            
            // 显示修改后的汇编
            console.log("[修改后的汇编]");
            console.log("=".repeat(70));
            console.log(hexdump(get_number, {
                length: 96,
                header: true,
                ansi: true
            }));
            console.log("");
            
        } catch (e) {
            console.log("[-] Patch 失败: " + e.message);
            console.log("\n尝试使用 Interceptor.replace() 代替...\n");
            
            Interceptor.replace(get_number, new NativeCallback(function() {
                return 42;
            }, 'int', []));
            
            console.log("✓ 使用 Interceptor.replace() 成功\n");
        }
        
    } else if (arch === 'arm') {
        console.log("[Patch] 使用 ARM32 汇编");
        console.log("=".repeat(70));
        console.log("将函数开头替换为：");
        console.log("  mov r0, #42");
        console.log("  bx lr");
        console.log("");
        
        try {
            Memory.patchCode(get_number, 8, function(code) {
                var writer = new ArmWriter(code, { pc: get_number });
                
                // mov r0, #42
                writer.putMovRegU8('r0', 42);
                
                // bx lr (返回)
                writer.putBxReg('lr');
                
                writer.flush();
                
                console.log("✓ Patch 成功！\n");
            });
            
            // 显示修改后的汇编
            console.log("[修改后的汇编]");
            console.log("=".repeat(70));
            console.log(hexdump(get_number, {
                length: 64,
                header: true,
                ansi: true
            }));
            console.log("");
            
        } catch (e) {
            console.log("[-] Patch 失败: " + e.message);
            console.log("\n尝试使用 Interceptor.replace() 代替...\n");
            
            Interceptor.replace(get_number, new NativeCallback(function() {
                return 42;
            }, 'int', []));
            
            console.log("✓ 使用 Interceptor.replace() 成功\n");
        }
        
    } else {
        console.log("[-] 不支持的架构: " + arch);
        console.log("使用 Interceptor.replace() 代替...\n");
        
        Interceptor.replace(get_number, new NativeCallback(function() {
            return 42;
        }, 'int', []));
        
        console.log("✓ 使用 Interceptor.replace() 成功\n");
    }
    
    // 测试验证
    console.log("=".repeat(70));
    console.log("[测试验证]");
    console.log("=".repeat(70) + "\n");
    
    var get_number_func = new NativeFunction(get_number, 'int', []);
    
    console.log("连续调用 15 次：\n");
    var all42 = true;
    
    for (var i = 0; i < 15; i++) {
        var result = get_number_func();
        var status = (result === 42) ? "✓" : "✗";
        
        if (result !== 42) {
            all42 = false;
        }
        
        console.log("  [" + (i + 1).toString().padStart(2) + "] " + status + " 返回值: " + result);
    }
    
    console.log("");
    
    if (all42) {
        console.log("🎉🎉🎉 完美！所有调用都返回 42！");
    } else {
        console.log("⚠️  有些调用不是 42，Patch 可能不完整");
    }
    
    console.log("\n" + "=".repeat(70));
    console.log("[✓] 内存 Patch 完成！");
    console.log("=".repeat(70) + "\n");
}

// Hook Java 层验证
Java.perform(function() {
    try {
        var MainActivity = Java.use("cn.binary.frida.MainActivity");
        
        MainActivity.GetNumber.implementation = function() {
            var result = this.GetNumber();
            
            console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            console.log("[Java 调用] MainActivity.GetNumber()");
            console.log("  Native 返回: " + result);
            
            if (result === 42) {
                console.log("  状态: ✓ 成功！返回 42");
            } else {
                console.log("  状态: ✗ 失败，返回 " + result);
            }
            console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            
            return result;
        };
        
        console.log("[✓] Java Hook 已设置\n");
        
    } catch (e) {
        console.log("[-] Java Hook 失败: " + e);
    }
});

