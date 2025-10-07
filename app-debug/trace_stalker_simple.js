// 🔍 Frida Stalker - 指令流追踪（简洁版）
// 适配 Frida 17.x

console.log("[🔍] Frida Stalker - 指令流追踪\n");

var moduleName = "libfrida.so";
var targetFunction = "get_number";  // 要追踪的函数

// 查找目标函数
var module = Process.findModuleByName(moduleName);
if (!module) {
    console.log("[-] 未找到模块: " + moduleName);
} else {
    console.log("[✓] 模块: " + moduleName);
    console.log("  基地址: " + module.base);
    console.log("");
    
    // 查找函数
    var targetAddr = null;
    var exports = Module.enumerateExports(moduleName);
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf(targetFunction) !== -1) {
            targetAddr = exp.address;
            console.log("[✓] 找到函数: " + exp.name);
            console.log("  地址: " + targetAddr);
            console.log("");
        }
    });
    
    if (targetAddr) {
        console.log("=".repeat(70));
        console.log("开始追踪指令流...");
        console.log("=".repeat(70) + "\n");
        
        var instructionCount = 0;
        var maxInstructions = 100;  // 限制显示前 100 条指令
        
        // Hook 函数入口
        Interceptor.attach(targetAddr, {
            onEnter: function(args) {
                console.log("[函数入口] " + targetFunction + " 被调用\n");
                
                // 开始追踪当前线程
                Stalker.follow(this.threadId, {
                    
                    // events 选项：指定要追踪的事件
                    events: {
                        call: false,    // 不追踪 call 指令
                        ret: false,     // 不追踪 ret 指令
                        exec: true,     // 追踪每条指令执行
                        block: false,   // 不追踪基本块
                        compile: false  // 不追踪编译事件
                    },
                    
                    // onReceive：接收追踪事件
                    onReceive: function(events) {
                        // events 是一个缓冲区，包含所有追踪的指令
                        var parsedEvents = Stalker.parse(events);
                        
                        parsedEvents.forEach(function(event) {
                            if (instructionCount < maxInstructions) {
                                // 显示指令地址和偏移
                                var offset = event[1].sub(module.base);
                                console.log(
                                    "[" + instructionCount.toString().padStart(4) + "] " +
                                    event[1] + " (+" + offset + ")"
                                );
                                instructionCount++;
                            }
                        });
                    }
                });
                
                this.stalking = true;
            },
            
            onLeave: function(retval) {
                if (this.stalking) {
                    // 停止追踪
                    Stalker.unfollow(this.threadId);
                    Stalker.flush();
                    
                    console.log("\n" + "=".repeat(70));
                    console.log("[函数退出] 返回值: " + retval);
                    console.log("总指令数: " + instructionCount + " 条");
                    console.log("=".repeat(70) + "\n");
                    
                    instructionCount = 0;
                }
            }
        });
        
        console.log("[✓] Stalker 已设置，等待函数调用...\n");
        
    } else {
        console.log("[-] 未找到目标函数");
    }
}

