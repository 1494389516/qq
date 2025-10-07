// 第十七关 - Patch Native 函数（高级版）
// 适配 Frida 17.x

console.log("[★] 第十七关：Patch Native 函数 - 高级版");
console.log("=".repeat(70) + "\n");

var moduleName = "libfrida.so";
var base = Module.findBaseAddress(moduleName);

if (!base) {
    console.log("[-] 未找到模块");
} else {
    console.log("[✓] " + moduleName);
    console.log("  基地址: " + base + "\n");
    
    // ═════════════════════════════════════════════════════════════
    // 查找目标函数
    // ═════════════════════════════════════════════════════════════
    
    console.log("[步骤 1] 查找 get_number 函数");
    console.log("─".repeat(70) + "\n");
    
    var exports = Module.enumerateExports(moduleName);
    var getNumberAddr = null;
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf("get_number") !== -1 || 
            exp.name.indexOf("GetNumber") !== -1) {
            
            getNumberAddr = exp.address;
            
            console.log("[找到] " + exp.name);
            console.log("  地址: " + exp.address);
            console.log("  偏移: 0x" + exp.address.sub(base).toString(16));
            console.log("");
        }
    });
    
    if (!getNumberAddr) {
        console.log("[-] 未找到 get_number 函数\n");
    } else {
        
        // ═════════════════════════════════════════════════════════════
        // 方法 1：使用 Interceptor.replace
        // ═════════════════════════════════════════════════════════════
        
        console.log("[方法 1] 使用 Interceptor.replace()");
        console.log("═".repeat(70) + "\n");
        
        console.log("创建新函数，返回固定值 42...");
        
        var newFunc = new NativeCallback(function() {
            console.log("  [Patch] 新函数被调用，返回 42");
            return 42;
        }, 'int', []);
        
        Interceptor.replace(getNumberAddr, newFunc);
        console.log("[✓] Patch 完成（方法 1）\n");
        
        // ═════════════════════════════════════════════════════════════
        // 方法 2：使用 Memory.patchCode
        // ═════════════════════════════════════════════════════════════
        
        console.log("[方法 2] 使用 Memory.patchCode()");
        console.log("═".repeat(70) + "\n");
        
        // 先显示原始代码
        console.log("原始代码:");
        console.log(hexdump(getNumberAddr, {
            length: 32,
            header: true,
            ansi: true
        }));
        
        // 备份原始代码
        var originalCode = getNumberAddr.readByteArray(32);
        
        // 根据架构选择指令
        if (Process.arch === 'arm64') {
            console.log("\n[ARM64 Patch]");
            console.log("─".repeat(70));
            
            Memory.patchCode(getNumberAddr, 8, function(code) {
                var writer = new Arm64Writer(code, { pc: getNumberAddr });
                
                // mov w0, #42    ; 返回值 42
                writer.putInstruction(0xd2800540);  // mov w0, #42
                
                // ret            ; 返回
                writer.putInstruction(0xd65f03c0);  // ret
                
                writer.flush();
            });
            
            console.log("  指令 1: mov w0, #42    ; w0 = 42");
            console.log("  指令 2: ret            ; 返回");
            console.log("  ✓ 已写入 ARM64 指令\n");
            
        } else if (Process.arch === 'arm') {
            console.log("\n[ARM Patch]");
            console.log("─".repeat(70));
            
            Memory.patchCode(getNumberAddr, 8, function(code) {
                var writer = new ArmWriter(code, { pc: getNumberAddr });
                
                // mov r0, #42    ; 返回值 42
                writer.putInstruction(0xe3a0002a);  // mov r0, #42
                
                // bx lr          ; 返回
                writer.putInstruction(0xe12fff1e);  // bx lr
                
                writer.flush();
            });
            
            console.log("  指令 1: mov r0, #42    ; r0 = 42");
            console.log("  指令 2: bx lr          ; 返回");
            console.log("  ✓ 已写入 ARM 指令\n");
            
        } else if (Process.arch === 'x64') {
            console.log("\n[x64 Patch]");
            console.log("─".repeat(70));
            
            Memory.patchCode(getNumberAddr, 16, function(code) {
                var writer = new X86Writer(code, { pc: getNumberAddr });
                
                // mov eax, 42    ; 返回值 42
                writer.putMovRegU32('eax', 42);
                
                // ret            ; 返回
                writer.putRet();
                
                writer.flush();
            });
            
            console.log("  指令 1: mov eax, 42    ; eax = 42");
            console.log("  指令 2: ret            ; 返回");
            console.log("  ✓ 已写入 x64 指令\n");
        }
        
        // 显示 Patch 后的代码
        console.log("Patch 后的代码:");
        console.log(hexdump(getNumberAddr, {
            length: 32,
            header: true,
            ansi: true
        }));
        
        console.log("[✓] Patch 完成（方法 2）\n");
        
        // ═════════════════════════════════════════════════════════════
        // 方法 3：使用 Interceptor.attach 修改返回值
        // ═════════════════════════════════════════════════════════════
        
        console.log("[方法 3] 使用 Interceptor.attach() 修改返回值");
        console.log("═".repeat(70) + "\n");
        
        // 注意：此时函数已经被方法1或2 patch了，这只是演示
        // 在实际使用时，选择一种方法即可
        
        Interceptor.attach(getNumberAddr, {
            onLeave: function(retval) {
                var original = retval.toInt32();
                console.log("  [Hook] 原返回值: " + original);
                
                retval.replace(42);
                console.log("  [Hook] 修改为: 42\n");
            }
        });
        
        console.log("[✓] Hook 完成（方法 3）\n");
        
        // ═════════════════════════════════════════════════════════════
        // Hook Java 层调用
        // ═════════════════════════════════════════════════════════════
        
        console.log("[步骤 2] Hook Java 层调用");
        console.log("─".repeat(70) + "\n");
        
        Java.perform(function() {
            try {
                var MainActivity = Java.use("cn.binary.frida.MainActivity");
                
                MainActivity.GetNumber.implementation = function() {
                    console.log("[Java Hook] GetNumber() 被调用");
                    
                    var result = this.GetNumber();
                    
                    console.log("  Native 返回: " + result);
                    console.log("  预期返回: 42");
                    
                    if (result === 42) {
                        console.log("  ✓ Patch 生效！\n");
                    } else {
                        console.log("  ✗ Patch 未生效\n");
                    }
                    
                    return result;
                };
                
                console.log("[✓] Java Hook 完成\n");
                
            } catch (e) {
                console.log("[-] Java Hook 失败: " + e + "\n");
            }
        });
        
        // ═════════════════════════════════════════════════════════════
        // 测试 Patch 效果
        // ═════════════════════════════════════════════════════════════
        
        console.log("[步骤 3] 测试 Patch 效果");
        console.log("═".repeat(70) + "\n");
        
        setTimeout(function() {
            console.log("[直接调用 Native 函数测试]");
            
            var testFunc = new NativeFunction(getNumberAddr, 'int', []);
            
            for (var i = 0; i < 5; i++) {
                var result = testFunc();
                console.log("  测试 #" + (i + 1) + ": " + result + 
                           (result === 42 ? " ✓" : " ✗"));
            }
            
            console.log("\n" + "═".repeat(70) + "\n");
        }, 1000);
        
        // ═════════════════════════════════════════════════════════════
        // 对比不同方法
        // ═════════════════════════════════════════════════════════════
        
        console.log("[三种 Patch 方法对比]");
        console.log("═".repeat(70));
        console.log("");
        console.log("方法 1: Interceptor.replace()");
        console.log("  优点: 简单易用，不需要写汇编");
        console.log("  缺点: 有性能开销，可能被检测");
        console.log("  适用: 快速测试，动态替换");
        console.log("");
        console.log("方法 2: Memory.patchCode()");
        console.log("  优点: 性能最佳，直接修改机器码");
        console.log("  缺点: 需要了解汇编，永久修改");
        console.log("  适用: 生产环境，绕过检测");
        console.log("");
        console.log("方法 3: Interceptor.attach() + retval.replace()");
        console.log("  优点: 灵活，可以动态决定返回值");
        console.log("  缺点: 有性能开销，每次调用都触发");
        console.log("  适用: 条件判断，动态修改");
        console.log("");
        console.log("═".repeat(70) + "\n");
        
        // ═════════════════════════════════════════════════════════════
        // 恢复功能（可选）
        // ═════════════════════════════════════════════════════════════
        
        console.log("[恢复功能]");
        console.log("─".repeat(70));
        console.log("原始代码已备份到 originalCode");
        console.log("如需恢复，运行:");
        console.log("  Memory.protect(getNumberAddr, 32, 'rwx');");
        console.log("  getNumberAddr.writeByteArray(originalCode);");
        console.log("═".repeat(70) + "\n");
        
        console.log("[✓] 所有 Patch 方法演示完成\n");
    }
    
    // ═════════════════════════════════════════════════════════════
    // 说明
    // ═════════════════════════════════════════════════════════════
    
    console.log("💡 什么是 Patch?");
    console.log("═".repeat(70));
    console.log("Patch 是指直接修改程序的机器码，改变其行为。");
    console.log("");
    console.log("常见应用场景:");
    console.log("  • 修改返回值（如破解会员验证）");
    console.log("  • 跳过检测逻辑（如去除广告）");
    console.log("  • 修改算法逻辑（如修改游戏数值）");
    console.log("  • 绕过混淆和加固");
    console.log("");
    console.log("注意事项:");
    console.log("  ⚠️  需要对应架构的汇编知识");
    console.log("  ⚠️  可能违反软件许可协议");
    console.log("  ⚠️  仅用于学习和研究");
    console.log("═".repeat(70) + "\n");
}


