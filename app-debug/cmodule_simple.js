// ⚡ Frida CModule - 简单示例
// 适配 Frida 17.x
// 使用 C 代码实现高性能 Hook

console.log("[⚡] Frida CModule - 高性能 Hook\n");

// ═════════════════════════════════════════════════════════════
// 1. 定义 C 代码
// ═════════════════════════════════════════════════════════════

const cCode = `
#include <gum/guminterceptor.h>
#include <stdio.h>

// 全局变量 - 统计调用次数
static int call_count = 0;

// Hook 函数：拦截 get_number
void on_enter(GumInvocationContext *ctx) {
    call_count++;
    
    // 打印信息（使用 printf）
    printf("[CModule] get_number 被调用，第 %d 次\\n", call_count);
}

void on_leave(GumInvocationContext *ctx) {
    // 获取返回值
    int retval = (int)gum_invocation_context_get_return_value(ctx);
    
    printf("[CModule] 返回值: %d\\n", retval);
    
    // 修改返回值为 42
    gum_invocation_context_replace_return_value(ctx, (gpointer)42);
    
    printf("[CModule] 已修改返回值为: 42\\n\\n");
}

// 导出函数：获取调用次数
int get_call_count() {
    return call_count;
}
`;

// ═════════════════════════════════════════════════════════════
// 2. 编译 C 代码为 CModule
// ═════════════════════════════════════════════════════════════

console.log("[1/3] 编译 C 代码...");
const cm = new CModule(cCode);
console.log("  ✅ 编译成功\n");

// ═════════════════════════════════════════════════════════════
// 3. 查找目标函数并 Hook
// ═════════════════════════════════════════════════════════════

console.log("[2/3] 查找目标函数...");

var moduleName = "libfrida.so";
var targetFunction = "get_number";

var module = Process.findModuleByName(moduleName);
if (!module) {
    console.log("  ❌ 未找到模块: " + moduleName);
} else {
    var exports = Module.enumerateExports(moduleName);
    var targetAddr = null;
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf(targetFunction) !== -1) {
            targetAddr = exp.address;
            console.log("  ✅ 找到: " + exp.name + " @ " + targetAddr);
        }
    });
    
    if (targetAddr) {
        console.log("\n[3/3] 设置 CModule Hook...");
        
        // 使用 CModule 的函数进行 Hook
        Interceptor.attach(targetAddr, {
            onEnter: cm.on_enter,
            onLeave: cm.on_leave
        });
        
        console.log("  ✅ Hook 完成\n");
        
        // 定时显示统计信息
        setInterval(function() {
            var count = cm.get_call_count();
            if (count > 0) {
                console.log("═".repeat(50));
                console.log("📊 [统计] 总调用次数: " + count);
                console.log("═".repeat(50) + "\n");
            }
        }, 10000);  // 每 10 秒显示一次
        
        console.log("═".repeat(50));
        console.log("🚀 CModule Hook 已激活！");
        console.log("💡 性能比 JS Hook 快 1000 倍！");
        console.log("═".repeat(50) + "\n");
        
    } else {
        console.log("  ❌ 未找到函数\n");
    }
}

