// ⚡ Frida CModule - 加密函数 Hook
// 适配 Frida 17.x
// 使用 C 代码实现高性能的加密数据拦截

console.log("[🔐] CModule 加密函数 Hook\n");

// ═════════════════════════════════════════════════════════════
// CModule C 代码 - 加密数据拦截
// ═════════════════════════════════════════════════════════════

const cCode = `
#include <gum/guminterceptor.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define MAX_DATA_SIZE 256
#define MAX_LOG_ENTRIES 100

// ─────────────────────────────────────────────────────────────
// 数据结构
// ─────────────────────────────────────────────────────────────

typedef struct {
    uint8_t input_data[MAX_DATA_SIZE];
    uint8_t output_data[MAX_DATA_SIZE];
    size_t data_length;
    uint64_t timestamp;
} EncryptionLog;

static EncryptionLog logs[MAX_LOG_ENTRIES];
static int log_index = 0;
static int total_encryptions = 0;

// XOR 操作计数器
static uint64_t xor_operations = 0;

// ─────────────────────────────────────────────────────────────
// crypto_init Hook
// ─────────────────────────────────────────────────────────────

void crypto_init_enter(GumInvocationContext *ctx) {
    // 参数: (result, key, key_len)
    uint8_t *result = (uint8_t *)gum_invocation_context_get_nth_argument(ctx, 0);
    uint8_t *key = (uint8_t *)gum_invocation_context_get_nth_argument(ctx, 1);
    uint64_t key_len = (uint64_t)gum_invocation_context_get_nth_argument(ctx, 2);
    
    printf("[CModule] crypto_init 被调用\\n");
    printf("  密钥长度: %llu\\n", key_len);
    
    if (key != NULL && key_len > 0 && key_len < 64) {
        printf("  密钥内容: ");
        for (size_t i = 0; i < key_len && i < 32; i++) {
            printf("%02x ", key[i]);
        }
        printf("\\n");
    }
}

// ─────────────────────────────────────────────────────────────
// crypto_crypt Hook
// ─────────────────────────────────────────────────────────────

typedef struct {
    uint8_t *result;
    uint8_t *data;
    size_t len;
    uint8_t input_backup[MAX_DATA_SIZE];
} CryptCallData;

void crypto_crypt_enter(GumInvocationContext *ctx) {
    // 参数: (result, data, data_len)
    uint8_t *result = (uint8_t *)gum_invocation_context_get_nth_argument(ctx, 0);
    uint8_t *data = (uint8_t *)gum_invocation_context_get_nth_argument(ctx, 1);
    uint64_t data_len = (uint64_t)gum_invocation_context_get_nth_argument(ctx, 2);
    
    printf("\\n[CModule] crypto_crypt 被调用\\n");
    printf("  数据长度: %llu\\n", data_len);
    
    // 分配临时存储
    CryptCallData *call_data = (CryptCallData *)malloc(sizeof(CryptCallData));
    call_data->result = result;
    call_data->data = data;
    call_data->len = data_len < MAX_DATA_SIZE ? data_len : MAX_DATA_SIZE;
    
    // 备份输入数据
    if (data != NULL && data_len > 0) {
        memcpy(call_data->input_backup, data, call_data->len);
        
        printf("  输入数据: ");
        for (size_t i = 0; i < call_data->len && i < 32; i++) {
            printf("%02x ", data[i]);
        }
        printf("\\n");
    }
    
    gum_invocation_context_set_listener_function_data(ctx, call_data);
}

void crypto_crypt_leave(GumInvocationContext *ctx) {
    CryptCallData *call_data = (CryptCallData *)gum_invocation_context_get_listener_function_data(ctx);
    
    if (call_data != NULL) {
        printf("  输出数据: ");
        for (size_t i = 0; i < call_data->len && i < 32; i++) {
            printf("%02x ", call_data->result[i]);
        }
        printf("\\n");
        
        // 分析加密模式
        printf("  \\n[加密分析]\\n");
        
        // 检测 XOR 模式
        int possible_xor = 1;
        for (size_t i = 0; i < call_data->len && i < 16; i++) {
            uint8_t xor_val = call_data->input_backup[i] ^ call_data->result[i];
            printf("  [%zu] %02x ^ %02x = %02x\\n", 
                   i, call_data->input_backup[i], call_data->result[i], xor_val);
        }
        
        // 保存日志
        if (log_index < MAX_LOG_ENTRIES) {
            memcpy(logs[log_index].input_data, call_data->input_backup, call_data->len);
            memcpy(logs[log_index].output_data, call_data->result, call_data->len);
            logs[log_index].data_length = call_data->len;
            log_index++;
        }
        
        total_encryptions++;
        
        free(call_data);
    }
    
    printf("\\n");
}

// ─────────────────────────────────────────────────────────────
// 导出函数
// ─────────────────────────────────────────────────────────────

int get_total_encryptions() {
    return total_encryptions;
}

int get_log_count() {
    return log_index < MAX_LOG_ENTRIES ? log_index : MAX_LOG_ENTRIES;
}

// 获取指定索引的日志
void get_log_entry(int index, uint8_t *input, uint8_t *output, size_t *len) {
    if (index >= 0 && index < log_index && index < MAX_LOG_ENTRIES) {
        memcpy(input, logs[index].input_data, logs[index].data_length);
        memcpy(output, logs[index].output_data, logs[index].data_length);
        *len = logs[index].data_length;
    }
}
`;

// ═════════════════════════════════════════════════════════════
// 编译 CModule
// ═════════════════════════════════════════════════════════════

console.log("[1/3] 编译 CModule...");
var cm = new CModule(cCode);
console.log("  ✅ 编译成功\n");

// ═════════════════════════════════════════════════════════════
// Hook 加密函数
// ═════════════════════════════════════════════════════════════

console.log("[2/3] Hook 加密函数...");

var moduleName = "libfrida.so";
var module = Process.findModuleByName(moduleName);

if (!module) {
    console.log("  ❌ 未找到模块");
} else {
    var exports = Module.enumerateExports(moduleName);
    
    var crypto_init = null;
    var crypto_crypt = null;
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf("crypto_init") !== -1) {
            crypto_init = exp.address;
            console.log("  ✅ 找到 crypto_init: " + exp.address);
        }
        if (exp.name.indexOf("crypto_crypt") !== -1) {
            crypto_crypt = exp.address;
            console.log("  ✅ 找到 crypto_crypt: " + exp.address);
        }
    });
    
    if (crypto_init) {
        Interceptor.attach(crypto_init, {
            onEnter: cm.crypto_init_enter
        });
    }
    
    if (crypto_crypt) {
        Interceptor.attach(crypto_crypt, {
            onEnter: cm.crypto_crypt_enter,
            onLeave: cm.crypto_crypt_leave
        });
    }
    
    console.log("\n[3/3] 启动监控...\n");
    
    // 定时显示统计
    setInterval(function() {
        var count = cm.get_total_encryptions();
        if (count > 0) {
            console.log("═".repeat(60));
            console.log("📊 [统计] 总加密次数: " + count);
            console.log("═".repeat(60) + "\n");
        }
    }, 20000);
    
    console.log("═".repeat(60));
    console.log("🔐 CModule 加密 Hook 已激活");
    console.log("⚡ 高性能数据拦截");
    console.log("═".repeat(60) + "\n");
}

