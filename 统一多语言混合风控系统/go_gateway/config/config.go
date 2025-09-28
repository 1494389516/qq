// 配置管理模块

package config

import (
	"fmt"
	"time"

	"github.com/spf13/viper"
)

// Config 应用程序配置结构
type Config struct {
	Server      ServerConfig      `mapstructure:"server"`
	Database    DatabaseConfig    `mapstructure:"database"`
	Redis       RedisConfig       `mapstructure:"redis"`
	Log         LogConfig         `mapstructure:"log"`
	RateLimit   RateLimitConfig   `mapstructure:"rate_limit"`
	Detection   DetectionConfig   `mapstructure:"detection"`
	Monitoring  MonitoringConfig  `mapstructure:"monitoring"`
	Security    SecurityConfig    `mapstructure:"security"`
}

// ServerConfig HTTP服务器配置
type ServerConfig struct {
	Port         int    `mapstructure:"port"`
	Mode         string `mapstructure:"mode"`
	ReadTimeout  int    `mapstructure:"read_timeout"`
	WriteTimeout int    `mapstructure:"write_timeout"`
	IdleTimeout  int    `mapstructure:"idle_timeout"`
}

// DatabaseConfig 数据库配置
type DatabaseConfig struct {
	Host     string `mapstructure:"host"`
	Port     int    `mapstructure:"port"`
	Username string `mapstructure:"username"`
	Password string `mapstructure:"password"`
	Database string `mapstructure:"database"`
	SSLMode  string `mapstructure:"ssl_mode"`
}

// RedisConfig Redis配置
type RedisConfig struct {
	Host     string `mapstructure:"host"`
	Port     int    `mapstructure:"port"`
	Password string `mapstructure:"password"`
	DB       int    `mapstructure:"db"`
}

// LogConfig 日志配置
type LogConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"`
	Output string `mapstructure:"output"`
}

// RateLimitConfig 限流配置
type RateLimitConfig struct {
	Enabled bool          `mapstructure:"enabled"`
	Rate    int           `mapstructure:"rate"`
	Burst   int           `mapstructure:"burst"`
	Window  time.Duration `mapstructure:"window"`
}

// DetectionConfig VPN检测配置
type DetectionConfig struct {
	ModelPath           string  `mapstructure:"model_path"`
	ConfidenceThreshold float64 `mapstructure:"confidence_threshold"`
	BatchSize           int     `mapstructure:"batch_size"`
	WindowSizeMs        int     `mapstructure:"window_size_ms"`
	StepSizeMs          int     `mapstructure:"step_size_ms"`
}

// MonitoringConfig 监控配置
type MonitoringConfig struct {
	Enabled           bool   `mapstructure:"enabled"`
	MetricsPath       string `mapstructure:"metrics_path"`
	AlertWebhookURL   string `mapstructure:"alert_webhook_url"`
	PerformanceTarget struct {
		Latency    time.Duration `mapstructure:"latency"`
		Throughput float64       `mapstructure:"throughput"`
		Accuracy   float64       `mapstructure:"accuracy"`
	} `mapstructure:"performance_target"`
}

// SecurityConfig 安全配置
type SecurityConfig struct {
	JWTSecret    string        `mapstructure:"jwt_secret"`
	TokenExpiry  time.Duration `mapstructure:"token_expiry"`
	EnableHTTPS  bool          `mapstructure:"enable_https"`
	CertFile     string        `mapstructure:"cert_file"`
	KeyFile      string        `mapstructure:"key_file"`
	TrustedProxy []string      `mapstructure:"trusted_proxy"`
}

// Load 加载配置
func Load() (*Config, error) {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("./configs/")
	viper.AddConfigPath("/etc/vpn-detection/")

	// 设置环境变量前缀
	viper.SetEnvPrefix("VPN_DETECTION")
	viper.AutomaticEnv()

	// 设置默认值
	setDefaults()

	// 读取配置文件
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			// 配置文件未找到，使用默认配置
			fmt.Println("配置文件未找到，使用默认配置")
		} else {
			return nil, fmt.Errorf("读取配置文件错误: %w", err)
		}
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("解析配置错误: %w", err)
	}

	return &config, nil
}

// setDefaults 设置默认配置值
func setDefaults() {
	// 服务器默认配置
	viper.SetDefault("server.port", 8080)
	viper.SetDefault("server.mode", "debug")
	viper.SetDefault("server.read_timeout", 30)
	viper.SetDefault("server.write_timeout", 30)
	viper.SetDefault("server.idle_timeout", 120)

	// 数据库默认配置
	viper.SetDefault("database.host", "localhost")
	viper.SetDefault("database.port", 5432)
	viper.SetDefault("database.username", "vpn_detection")
	viper.SetDefault("database.password", "password")
	viper.SetDefault("database.database", "vpn_detection")
	viper.SetDefault("database.ssl_mode", "disable")

	// Redis默认配置
	viper.SetDefault("redis.host", "localhost")
	viper.SetDefault("redis.port", 6379)
	viper.SetDefault("redis.password", "")
	viper.SetDefault("redis.db", 0)

	// 日志默认配置
	viper.SetDefault("log.level", "info")
	viper.SetDefault("log.format", "json")
	viper.SetDefault("log.output", "stdout")

	// 限流默认配置
	viper.SetDefault("rate_limit.enabled", true)
	viper.SetDefault("rate_limit.rate", 100)
	viper.SetDefault("rate_limit.burst", 200)
	viper.SetDefault("rate_limit.window", "1m")

	// 检测默认配置
	viper.SetDefault("detection.model_path", "./models/vpn_detection.bin")
	viper.SetDefault("detection.confidence_threshold", 0.7)
	viper.SetDefault("detection.batch_size", 100)
	viper.SetDefault("detection.window_size_ms", 5000)
	viper.SetDefault("detection.step_size_ms", 2000)

	// 监控默认配置
	viper.SetDefault("monitoring.enabled", true)
	viper.SetDefault("monitoring.metrics_path", "/metrics")
	viper.SetDefault("monitoring.performance_target.latency", "100ms")
	viper.SetDefault("monitoring.performance_target.throughput", 10.0)
	viper.SetDefault("monitoring.performance_target.accuracy", 0.95)

	// 安全默认配置
	viper.SetDefault("security.jwt_secret", "default_secret_change_in_production")
	viper.SetDefault("security.token_expiry", "24h")
	viper.SetDefault("security.enable_https", false)
	viper.SetDefault("security.trusted_proxy", []string{"127.0.0.1"})
}

// Validate 验证配置
func (c *Config) Validate() error {
	if c.Server.Port <= 0 || c.Server.Port > 65535 {
		return fmt.Errorf("无效的服务器端口: %d", c.Server.Port)
	}

	if c.Detection.ConfidenceThreshold < 0 || c.Detection.ConfidenceThreshold > 1 {
		return fmt.Errorf("无效的置信度阈值: %f", c.Detection.ConfidenceThreshold)
	}

	if c.Detection.BatchSize <= 0 {
		return fmt.Errorf("无效的批处理大小: %d", c.Detection.BatchSize)
	}

	return nil
}

// String 返回配置的字符串表示(隐藏敏感信息)
func (c *Config) String() string {
	return fmt.Sprintf("Config{Server: %+v, Log: %+v, Detection: %+v}",
		c.Server, c.Log, c.Detection)
}