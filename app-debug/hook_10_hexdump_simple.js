// 第十关 - hexdump 打印内存（简洁版）

console.log("[★] 第十关：hexdump 打印 libc.so + 0x200 的 128 字节\n");

// 获取基地址
var base = Module.findBaseAddress("libc.so");
console.log("基地址: " + base);

// 计算目标地址
var target = base.add(0x200);
console.log("目标地址: " + target + "\n");

// Hexdump
console.log("=".repeat(60));
console.log(hexdump(target, {
    length: 128,
    header: true,
    ansi: true
}));
console.log("=".repeat(60));
