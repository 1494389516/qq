// 🔍 Frida Stalker - 高级指令流追踪
// 适配 Frida 17.x
// 包含指令反汇编、内存访问、分支追踪

console.log("╔═══════════════════════════════════════════════════════════════╗");
console.log("║          🔍 Frida Stalker 高级指令追踪系统               ║");
console.log("╚═══════════════════════════════════════════════════════════════╝\n");

var moduleName = "libfrida.so";
var targetFunction = "get_number";

// 配置选项
var config = {
    showDisassembly: true,      // 显示反汇编
    showMemoryAccess: false,    // 显示内存访问
    showBranches: true,         // 显示分支跳转
    maxInstructions: 200,       // 最多显示指令数
    saveToFile: false           // 是否保存到文件
};

var stats = {
    totalInstructions: 0,
    branches: 0,
    calls: 0,
    returns: 0,
    memoryReads: 0,
    memoryWrites: 0
};

// ═════════════════════════════════════════════════════════════
// 工具函数
// ═════════════════════════════════════════════════════════════

function formatAddress(addr, base) {
    var offset = addr.sub(base);
    return addr + " (模块+" + offset + ")";
}

function getInstructionInfo(address) {
    try {
        var instruction = Instruction.parse(address);
        return {
            mnemonic: instruction.mnemonic,
            opStr: instruction.opStr,
            size: instruction.size,
            toString: function() {
                return this.mnemonic + " " + this.opStr;
            }
        };
    } catch (e) {
        return {
            mnemonic: "???",
            opStr: "",
            size: 4,
            toString: function() { return "无法解析"; }
        };
    }
}

// ═════════════════════════════════════════════════════════════
// 主追踪逻辑
// ═════════════════════════════════════════════════════════════

var module = Process.findModuleByName(moduleName);
if (!module) {
    console.log("[-] 未找到模块: " + moduleName);
} else {
    console.log("[模块信息]");
    console.log("  名称: " + module.name);
    console.log("  基地址: " + module.base);
    console.log("  大小: " + module.size + " 字节");
    console.log("  路径: " + module.path);
    console.log("");
    
    // 查找目标函数
    var targetAddr = null;
    var targetName = "";
    var exports = Module.enumerateExports(moduleName);
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf(targetFunction) !== -1) {
            targetAddr = exp.address;
            targetName = exp.name;
        }
    });
    
    if (!targetAddr) {
        console.log("[-] 未找到函数: " + targetFunction);
    } else {
        console.log("[目标函数]");
        console.log("  名称: " + targetName);
        console.log("  地址: " + targetAddr);
        console.log("  偏移: 0x" + targetAddr.sub(module.base).toString(16));
        console.log("");
        
        console.log("[追踪配置]");
        console.log("  反汇编显示: " + (config.showDisassembly ? "✓" : "✗"));
        console.log("  内存访问追踪: " + (config.showMemoryAccess ? "✓" : "✗"));
        console.log("  分支追踪: " + (config.showBranches ? "✓" : "✗"));
        console.log("  最大指令数: " + config.maxInstructions);
        console.log("");
        
        // 设置 Hook
        Interceptor.attach(targetAddr, {
            onEnter: function(args) {
                console.log("═".repeat(70));
                console.log("[函数调用] " + targetName);
                console.log("═".repeat(70));
                console.log("时间: " + new Date().toLocaleTimeString());
                console.log("线程ID: " + this.threadId);
                console.log("");
                
                // 重置统计
                stats.totalInstructions = 0;
                stats.branches = 0;
                stats.calls = 0;
                stats.returns = 0;
                
                var instructionLog = [];
                
                // 开始追踪
                Stalker.follow(this.threadId, {
                    
                    // 使用 transform 可以更精确地控制追踪
                    transform: function(iterator) {
                        var instruction = iterator.next();
                        
                        do {
                            if (stats.totalInstructions < config.maxInstructions) {
                                
                                // 判断指令类型
                                var isCall = instruction.mnemonic.startsWith("bl") || 
                                           instruction.mnemonic === "call";
                                var isRet = instruction.mnemonic === "ret" || 
                                          instruction.mnemonic === "bx";
                                var isBranch = instruction.mnemonic.startsWith("b") && 
                                             !isCall && !isRet;
                                
                                // 统计
                                stats.totalInstructions++;
                                if (isCall) stats.calls++;
                                if (isRet) stats.returns++;
                                if (isBranch) stats.branches++;
                                
                                // 记录指令信息
                                var info = {
                                    index: stats.totalInstructions,
                                    address: instruction.address,
                                    offset: instruction.address.sub(module.base),
                                    mnemonic: instruction.mnemonic,
                                    opStr: instruction.opStr,
                                    isCall: isCall,
                                    isRet: isRet,
                                    isBranch: isBranch
                                };
                                
                                instructionLog.push(info);
                                
                                // 实时显示（可选）
                                if (config.showDisassembly) {
                                    var prefix = "";
                                    if (isCall) prefix = "[CALL] ";
                                    else if (isRet) prefix = "[RET]  ";
                                    else if (isBranch) prefix = "[BR]   ";
                                    else prefix = "       ";
                                    
                                    console.log(
                                        prefix +
                                        "[" + info.index.toString().padStart(4) + "] " +
                                        info.address + " : " +
                                        info.mnemonic.padEnd(8) + " " +
                                        info.opStr
                                    );
                                }
                            }
                            
                            // 保持指令不变
                            iterator.keep();
                            
                        } while ((instruction = iterator.next()) !== null);
                    }
                });
                
                this.instructionLog = instructionLog;
                this.stalking = true;
            },
            
            onLeave: function(retval) {
                if (this.stalking) {
                    // 停止追踪
                    Stalker.unfollow(this.threadId);
                    Stalker.flush();
                    
                    console.log("\n" + "═".repeat(70));
                    console.log("[函数返回]");
                    console.log("═".repeat(70));
                    console.log("返回值: " + retval);
                    console.log("");
                    
                    // 显示统计信息
                    console.log("[执行统计]");
                    console.log("─".repeat(70));
                    console.log("  总指令数: " + stats.totalInstructions + " 条");
                    console.log("  分支跳转: " + stats.branches + " 次");
                    console.log("  函数调用: " + stats.calls + " 次");
                    console.log("  函数返回: " + stats.returns + " 次");
                    console.log("");
                    
                    // 分析执行路径
                    if (this.instructionLog && this.instructionLog.length > 0) {
                        console.log("[执行路径分析]");
                        console.log("─".repeat(70));
                        
                        var firstAddr = this.instructionLog[0].address;
                        var lastAddr = this.instructionLog[this.instructionLog.length - 1].address;
                        
                        console.log("  起始地址: " + firstAddr);
                        console.log("  结束地址: " + lastAddr);
                        console.log("  地址跨度: 0x" + lastAddr.sub(firstAddr).toString(16) + " 字节");
                        console.log("");
                        
                        // 找出所有分支目标
                        var branches = this.instructionLog.filter(function(i) { 
                            return i.isBranch || i.isCall; 
                        });
                        
                        if (branches.length > 0) {
                            console.log("  关键跳转/调用:");
                            branches.slice(0, 10).forEach(function(b) {
                                console.log("    [" + b.index + "] " + 
                                          b.address + " : " + 
                                          b.mnemonic + " " + b.opStr);
                            });
                            
                            if (branches.length > 10) {
                                console.log("    ... 还有 " + (branches.length - 10) + " 个");
                            }
                        }
                    }
                    
                    console.log("═".repeat(70) + "\n");
                }
            }
        });
        
        console.log("✅ Stalker 追踪已设置");
        console.log("⏳ 等待函数 " + targetName + " 被调用...\n");
    }
}

