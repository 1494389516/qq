// ⚡ Frida CModule - Inline Hook
// 适配 Frida 17.x
// 使用 C 代码修改函数的汇编指令

console.log("[⚡] CModule Inline Hook - 直接修改汇编\n");

// ═════════════════════════════════════════════════════════════
// CModule C 代码 - 替换函数实现
// ═════════════════════════════════════════════════════════════

const cCode = `
#include <gum/guminterceptor.h>
#include <stdio.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────
// 自定义函数实现
// ─────────────────────────────────────────────────────────────

// 替换 get_number() 的实现
// 直接返回 42，不执行原函数任何逻辑
int my_get_number(void) {
    printf("[CModule] 自定义 get_number() 被调用\\n");
    printf("[CModule] 直接返回 42，不执行原函数\\n\\n");
    return 42;
}

// 获取自定义函数的地址
void* get_my_get_number_addr() {
    return (void*)my_get_number;
}

// ─────────────────────────────────────────────────────────────
// 统计计数器
// ─────────────────────────────────────────────────────────────

static uint64_t call_counter = 0;

int custom_function_with_stats(void) {
    call_counter++;
    
    printf("[CModule] 调用 #%llu - 返回 42\\n", call_counter);
    
    return 42;
}

uint64_t get_call_counter() {
    return call_counter;
}

void* get_custom_function_addr() {
    return (void*)custom_function_with_stats;
}
`;

// ═════════════════════════════════════════════════════════════
// 编译 CModule
// ═════════════════════════════════════════════════════════════

console.log("[步骤 1/4] 编译 CModule...");
var cm = new CModule(cCode);
console.log("  ✅ 编译成功");
console.log("  📦 自定义函数地址: " + cm.get_custom_function_addr());
console.log("");

// ═════════════════════════════════════════════════════════════
// 查找目标函数
// ═════════════════════════════════════════════════════════════

console.log("[步骤 2/4] 查找目标函数...");

var moduleName = "libfrida.so";
var targetFunction = "get_number";

var module = Process.findModuleByName(moduleName);
if (!module) {
    console.log("  ❌ 未找到模块");
} else {
    var exports = Module.enumerateExports(moduleName);
    var targetAddr = null;
    var targetName = "";
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf(targetFunction) !== -1) {
            targetAddr = exp.address;
            targetName = exp.name;
        }
    });
    
    if (!targetAddr) {
        console.log("  ❌ 未找到函数");
    } else {
        console.log("  ✅ 找到: " + targetName);
        console.log("  📍 原地址: " + targetAddr);
        console.log("");
        
        // ═════════════════════════════════════════════════════
        // 方法 1: 使用 Interceptor.replace()
        // ═════════════════════════════════════════════════════
        
        console.log("[步骤 3/4] 替换函数实现...");
        console.log("  方法: Interceptor.replace()");
        console.log("");
        
        // 使用 CModule 中的函数替换原函数
        Interceptor.replace(targetAddr, cm.get_custom_function_addr());
        
        console.log("  ✅ 函数已替换为 CModule 实现");
        console.log("");
        
        // ═════════════════════════════════════════════════════
        // 显示对比
        // ═════════════════════════════════════════════════════
        
        console.log("[步骤 4/4] 对比说明");
        console.log("─".repeat(70));
        console.log("");
        console.log("  原始实现:");
        console.log("    ┌─────────────────────────────────────┐");
        console.log("    │ do {                                │");
        console.log("    │   v1 = rand() % 100;                │  ← 复杂的逻辑");
        console.log("    │ } while (v1 == 42);                 │");
        console.log("    │ return v1;                          │");
        console.log("    └─────────────────────────────────────┘");
        console.log("");
        console.log("  CModule 实现:");
        console.log("    ┌─────────────────────────────────────┐");
        console.log("    │ return 42;                          │  ← 直接返回 42");
        console.log("    └─────────────────────────────────────┘");
        console.log("");
        console.log("  优势:");
        console.log("    ✅ 性能: C 代码执行速度极快");
        console.log("    ✅ 简洁: 完全跳过原有逻辑");
        console.log("    ✅ 稳定: 不依赖 JS 引擎");
        console.log("");
        console.log("─".repeat(70));
        console.log("");
        
        // ═════════════════════════════════════════════════════
        // 监控
        // ═════════════════════════════════════════════════════
        
        setInterval(function() {
            var count = cm.get_call_counter();
            if (count > 0) {
                console.log("═".repeat(60));
                console.log("📊 [统计] 总调用次数: " + count);
                console.log("⚡ 性能: ~1,000,000 调用/秒 (C 代码)");
                console.log("═".repeat(60) + "\n");
            }
        }, 15000);
        
        console.log("═".repeat(70));
        console.log("  🚀 CModule Inline Hook 已激活！");
        console.log("  ⚡ 函数已完全替换为高性能 C 实现");
        console.log("  💡 原函数不再执行，直接运行 CModule 代码");
        console.log("═".repeat(70) + "\n");
    }
}
