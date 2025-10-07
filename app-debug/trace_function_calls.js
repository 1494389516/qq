// 📞 Frida Trace - 函数调用追踪
// 适配 Frida 17.x
// 追踪模块内所有函数的调用关系

console.log("[📞] Frida Trace - 函数调用追踪\n");

var moduleName = "libfrida.so";
var maxCallDepth = 10;  // 最大调用深度
var currentDepth = 0;

console.log("[配置]");
console.log("  目标模块: " + moduleName);
console.log("  最大深度: " + maxCallDepth);
console.log("");

var module = Process.findModuleByName(moduleName);
if (!module) {
    console.log("[-] 未找到模块");
} else {
    console.log("[✓] 模块信息");
    console.log("  基地址: " + module.base);
    console.log("  大小: " + module.size + " 字节");
    console.log("");
    
    console.log("=".repeat(70));
    console.log("枚举所有导出函数并设置追踪...");
    console.log("=".repeat(70) + "\n");
    
    var exports = Module.enumerateExports(moduleName);
    var hookedCount = 0;
    
    exports.forEach(function(exp) {
        // 只 Hook 函数（type = "function"）
        if (exp.type === "function") {
            try {
                Interceptor.attach(exp.address, {
                    onEnter: function(args) {
                        currentDepth++;
                        
                        if (currentDepth <= maxCallDepth) {
                            var indent = "  ".repeat(currentDepth - 1);
                            var arrow = currentDepth > 1 ? "└─> " : "";
                            
                            console.log(
                                indent + arrow + 
                                "[" + currentDepth + "] " + 
                                exp.name + "()"
                            );
                            
                            // 显示参数（前4个）
                            if (args && args.length > 0) {
                                for (var i = 0; i < Math.min(4, args.length); i++) {
                                    console.log(
                                        indent + "    arg" + i + ": " + args[i]
                                    );
                                }
                            }
                        }
                        
                        this.depth = currentDepth;
                        this.name = exp.name;
                    },
                    
                    onLeave: function(retval) {
                        if (this.depth <= maxCallDepth) {
                            var indent = "  ".repeat(this.depth - 1);
                            console.log(
                                indent + "  ← " + this.name + " 返回: " + retval
                            );
                        }
                        
                        currentDepth--;
                    }
                });
                
                hookedCount++;
                
            } catch (e) {
                // Hook 失败，可能不是有效的函数入口
            }
        }
    });
    
    console.log("\n[✓] 已设置 " + hookedCount + " 个函数追踪器");
    console.log("⏳ 等待函数调用...\n");
    console.log("=".repeat(70));
    console.log("函数调用树:");
    console.log("=".repeat(70) + "\n");
}

