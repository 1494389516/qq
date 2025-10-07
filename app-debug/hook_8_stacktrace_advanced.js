// 第八关 - 打印调用栈（高级版）
// 适配 Frida 17.x

console.log("[★] 第八关：打印Java调用栈 - 高级版");
console.log("=".repeat(70) + "\n");

Java.perform(function() {
    
    console.log("[模块 1] Hook target() 方法");
    console.log("─".repeat(70) + "\n");
    
    var StackTrace = Java.use("cn.binary.frida.StackTrace");
    
    StackTrace.target.implementation = function() {
        console.log("\n" + "═".repeat(70));
        console.log("[target() 方法被调用]");
        console.log("═".repeat(70));
        
        // 获取调用栈
        var Thread = Java.use("java.lang.Thread");
        var Exception = Java.use("java.lang.Exception");
        
        var stack = Thread.currentThread().getStackTrace();
        
        console.log("\n[完整调用栈]");
        console.log("─".repeat(70));
        console.log("线程: " + Thread.currentThread().getName());
        console.log("线程ID: " + Thread.currentThread().getId());
        console.log("调用深度: " + stack.length);
        console.log("");
        
        // 显示所有栈帧
        for (var i = 0; i < stack.length; i++) {
            var element = stack[i];
            var className = element.getClassName();
            var methodName = element.getMethodName();
            var fileName = element.getFileName();
            var lineNumber = element.getLineNumber();
            
            // 过滤 Frida 相关的栈帧
            if (className.indexOf("frida") === -1 && 
                className.indexOf("Frida") === -1) {
                
                var indent = "  ";
                var arrow = i > 0 ? "↓ " : "• ";
                
                console.log(indent + arrow + "[" + i + "] " + className + "." + methodName + "()");
                
                if (fileName && lineNumber >= 0) {
                    console.log(indent + "   文件: " + fileName + ":" + lineNumber);
                }
                
                // 高亮显示关键方法
                if (methodName.indexOf("target") !== -1 || 
                    methodName.indexOf("sub") !== -1 ||
                    methodName.indexOf("top") !== -1) {
                    console.log(indent + "   ⭐ 关键方法");
                }
                
                console.log("");
            }
        }
        
        // 分析调用链
        console.log("─".repeat(70));
        console.log("[调用链分析]");
        console.log("─".repeat(70));
        
        var targetChain = [];
        for (var i = 0; i < stack.length; i++) {
            var methodName = stack[i].getMethodName();
            if (methodName.indexOf("sub") !== -1 || 
                methodName.indexOf("top") !== -1 ||
                methodName.indexOf("target") !== -1) {
                targetChain.push(methodName);
            }
        }
        
        if (targetChain.length > 0) {
            console.log("关键调用路径:");
            console.log("  " + targetChain.reverse().join(" → "));
            console.log("");
        }
        
        // 使用 Exception 获取更详细的栈信息
        try {
            var ex = Exception.$new("Stack trace");
            console.log("[Exception 栈信息]");
            console.log("─".repeat(70));
            
            var stackStr = ex.getStackTrace();
            for (var i = 0; i < Math.min(10, stackStr.length); i++) {
                console.log("  " + stackStr[i].toString());
            }
        } catch (e) {}
        
        console.log("\n" + "═".repeat(70) + "\n");
        
        return this.target();
    };
    
    // ═════════════════════════════════════════════════════════════
    // 模块 2：Hook 所有 sub 方法
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 2] Hook 所有 sub 方法");
    console.log("─".repeat(70));
    
    var callDepth = 0;
    
    for (var i = 1; i <= 13; i++) {
        try {
            (function(index) {
                var methodName = "sub" + index;
                StackTrace[methodName].implementation = function() {
                    callDepth++;
                    
                    var indent = "  ".repeat(callDepth);
                    console.log(indent + "→ [深度 " + callDepth + "] " + methodName + "()");
                    
                    var result = this[methodName]();
                    
                    console.log(indent + "← [深度 " + callDepth + "] " + methodName + "() 返回");
                    callDepth--;
                    
                    return result;
                };
            })(i);
        } catch (e) {}
    }
    
    console.log("  ✓ 已Hook sub1 到 sub13");
    console.log("");
    
    // ═════════════════════════════════════════════════════════════
    // 模块 3：Hook top 方法
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 3] Hook top() 方法");
    console.log("─".repeat(70));
    
    StackTrace.top.implementation = function() {
        console.log("\n" + "═".repeat(70));
        console.log("[top() 开始执行 - 调用链的起点]");
        console.log("═".repeat(70) + "\n");
        
        callDepth = 0;
        
        var result = this.top();
        
        console.log("\n" + "═".repeat(70));
        console.log("[top() 执行完成]");
        console.log("═".repeat(70) + "\n");
        
        return result;
    };
    
    console.log("  ✓ 已Hook top()\n");
    
    console.log("[✓] 所有 Hook 已设置完成\n");
    console.log("💡 提示：调用 top() 方法后，将显示完整的14层调用链\n");
});

