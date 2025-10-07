// 第八关 - 打印调用栈（简洁版）

Java.perform(function() {
    console.log("[★] 第八关：打印调用栈\n");
    
    var StackTrace = Java.use("cn.binary.frida.StackTrace");
    
    StackTrace.target.implementation = function() {
        console.log("\n" + "=".repeat(50));
        console.log("[target() 被调用 - 打印调用栈]");
        console.log("=".repeat(50) + "\n");
        
        // 获取调用栈
        var Thread = Java.use("java.lang.Thread");
        var stack = Thread.currentThread().getStackTrace();
        
        // 打印调用栈
        for (var i = 0; i < stack.length; i++) {
            var e = stack[i];
            console.log("  [" + i + "] " + e.getClassName() + "." + 
                       e.getMethodName() + ":" + e.getLineNumber());
        }
        
        // 提取并显示 StackTrace 的调用链
        console.log("\n[调用链]");
        var chain = [];
        for (var j = stack.length - 1; j >= 0; j--) {
            if (stack[j].getClassName().indexOf("StackTrace") !== -1) {
                chain.push(stack[j].getMethodName());
            }
        }
        console.log("  " + chain.join(" → "));
        
        console.log("\n" + "=".repeat(50) + "\n");
        
        return this.target();
    };
    
    console.log("[✓] Hook 完成\n");
});
