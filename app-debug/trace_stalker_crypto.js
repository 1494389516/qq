// 🔍 Frida Stalker - 追踪加密函数
// 适配 Frida 17.x
// 专门用于追踪加密算法的执行流程

console.log("[🔐] Stalker 加密函数追踪器\n");

var moduleName = "libfrida.so";

// 要追踪的加密函数
var targetFunctions = ["crypto_init", "crypto_crypt"];

console.log("[目标] 追踪加密函数的执行流程");
console.log("  模块: " + moduleName);
console.log("  函数: " + targetFunctions.join(", "));
console.log("");

var module = Process.findModuleByName(moduleName);
if (!module) {
    console.log("[-] 未找到模块");
} else {
    console.log("[✓] 模块加载完成");
    console.log("  基地址: " + module.base);
    console.log("");
    
    // 查找所有目标函数
    var exports = Module.enumerateExports(moduleName);
    var foundFunctions = [];
    
    exports.forEach(function(exp) {
        targetFunctions.forEach(function(targetName) {
            if (exp.name.indexOf(targetName) !== -1) {
                foundFunctions.push({
                    name: exp.name,
                    address: exp.address
                });
                console.log("[找到] " + exp.name + " @ " + exp.address);
            }
        });
    });
    
    if (foundFunctions.length === 0) {
        console.log("[-] 未找到目标函数");
    } else {
        console.log("\n" + "=".repeat(70));
        console.log("开始设置追踪...");
        console.log("=".repeat(70) + "\n");
        
        foundFunctions.forEach(function(func) {
            console.log("[设置追踪] " + func.name);
            
            Interceptor.attach(func.address, {
                onEnter: function(args) {
                    console.log("\n" + "═".repeat(70));
                    console.log("🔐 [加密函数调用] " + func.name);
                    console.log("═".repeat(70));
                    console.log("参数:");
                    console.log("  arg0 (result): " + args[0]);
                    console.log("  arg1 (data):   " + args[1]);
                    console.log("  arg2 (len):    " + args[2]);
                    
                    // 显示输入数据
                    try {
                        var len = parseInt(args[2]);
                        if (len > 0 && len < 256) {
                            console.log("\n[输入数据]");
                            console.log(hexdump(args[1], {
                                length: Math.min(len, 64),
                                header: false,
                                ansi: true
                            }));
                        }
                    } catch (e) {}
                    
                    console.log("\n[开始追踪指令流...]");
                    console.log("─".repeat(70) + "\n");
                    
                    var instructions = [];
                    var xorCount = 0;
                    var addCount = 0;
                    var loadCount = 0;
                    var storeCount = 0;
                    
                    // 开始追踪
                    Stalker.follow(this.threadId, {
                        transform: function(iterator) {
                            var instruction = iterator.next();
                            var count = 0;
                            
                            do {
                                if (count < 500) {  // 限制追踪的指令数
                                    var mnemonic = instruction.mnemonic.toLowerCase();
                                    
                                    // 统计关键操作
                                    if (mnemonic.includes("eor") || mnemonic.includes("xor")) {
                                        xorCount++;
                                        console.log("  [XOR] " + instruction.address + " : " + 
                                                  instruction.mnemonic + " " + instruction.opStr);
                                    }
                                    else if (mnemonic.includes("add") || mnemonic.includes("sub")) {
                                        addCount++;
                                    }
                                    else if (mnemonic.includes("ldr") || mnemonic.includes("load")) {
                                        loadCount++;
                                    }
                                    else if (mnemonic.includes("str") || mnemonic.includes("store")) {
                                        storeCount++;
                                    }
                                    
                                    // 记录特殊指令
                                    if (mnemonic.includes("eor") || 
                                        mnemonic.includes("xor") ||
                                        mnemonic.includes("ror") ||
                                        mnemonic.includes("rol") ||
                                        mnemonic.includes("aes")) {
                                        
                                        instructions.push({
                                            address: instruction.address,
                                            mnemonic: instruction.mnemonic,
                                            opStr: instruction.opStr
                                        });
                                    }
                                    
                                    count++;
                                }
                                
                                iterator.keep();
                                
                            } while ((instruction = iterator.next()) !== null);
                        }
                    });
                    
                    this.instructions = instructions;
                    this.xorCount = xorCount;
                    this.addCount = addCount;
                    this.loadCount = loadCount;
                    this.storeCount = storeCount;
                    this.resultPtr = args[0];
                    this.stalking = true;
                },
                
                onLeave: function(retval) {
                    if (this.stalking) {
                        Stalker.unfollow(this.threadId);
                        Stalker.flush();
                        
                        console.log("\n─".repeat(70));
                        console.log("[加密操作统计]");
                        console.log("─".repeat(70));
                        console.log("  XOR 操作: " + this.xorCount + " 次");
                        console.log("  加/减操作: " + this.addCount + " 次");
                        console.log("  内存加载: " + this.loadCount + " 次");
                        console.log("  内存存储: " + this.storeCount + " 次");
                        
                        if (this.instructions.length > 0) {
                            console.log("\n[关键加密指令]");
                            console.log("─".repeat(70));
                            this.instructions.slice(0, 20).forEach(function(inst, i) {
                                console.log("  [" + (i+1) + "] " + 
                                          inst.address + " : " + 
                                          inst.mnemonic + " " + inst.opStr);
                            });
                            
                            if (this.instructions.length > 20) {
                                console.log("  ... 还有 " + (this.instructions.length - 20) + " 条");
                            }
                        }
                        
                        // 显示加密结果
                        try {
                            console.log("\n[加密结果]");
                            console.log("─".repeat(70));
                            console.log(hexdump(this.resultPtr, {
                                length: 64,
                                header: false,
                                ansi: true
                            }));
                        } catch (e) {}
                        
                        console.log("\n═".repeat(70));
                        console.log("✅ 追踪完成");
                        console.log("═".repeat(70) + "\n");
                    }
                }
            });
        });
        
        console.log("\n[✓] 所有追踪器已设置完成");
        console.log("⏳ 等待加密函数被调用...\n");
    }
}

