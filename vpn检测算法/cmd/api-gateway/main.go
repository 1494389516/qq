// VPN检测系统 - API网关服务
// 基于Go语言实现高性能的RESTful API和WebSocket服务

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"

	"vpn-detection-system/internal/api"
	"vpn-detection-system/internal/config"
	"vpn-detection-system/internal/metrics"
	"vpn-detection-system/internal/middleware"
	"vpn-detection-system/internal/websocket"
)

// @title VPN检测系统 API
// @version 1.0
// @description 企业级VPN检测系统的RESTful API接口
// @termsOfService http://swagger.io/terms/

// @contact.name API Support
// @contact.url http://www.swagger.io/support
// @contact.email support@swagger.io

// @license.name Apache 2.0
// @license.url http://www.apache.org/licenses/LICENSE-2.0.html

// @host localhost:8080
// @BasePath /api/v1

func main() {
	// 初始化配置
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("加载配置失败: %v", err)
	}

	// 初始化日志
	setupLogger(cfg.Log.Level)

	// 初始化指标收集
	metrics.Init()

	// 设置Gin模式
	if cfg.Server.Mode == "release" {
		gin.SetMode(gin.ReleaseMode)
	}

	// 创建路由器
	router := setupRouter(cfg)

	// 创建HTTP服务器
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
		IdleTimeout:  time.Duration(cfg.Server.IdleTimeout) * time.Second,
	}

	// 启动服务器
	go func() {
		logrus.Infof("启动API网关服务，监听端口: %d", cfg.Server.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logrus.Fatalf("服务器启动失败: %v", err)
		}
	}()

	// 优雅关闭
	gracefulShutdown(server)
}

func setupLogger(level string) {
	logrus.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: time.RFC3339,
	})

	switch level {
	case "debug":
		logrus.SetLevel(logrus.DebugLevel)
	case "info":
		logrus.SetLevel(logrus.InfoLevel)
	case "warn":
		logrus.SetLevel(logrus.WarnLevel)
	case "error":
		logrus.SetLevel(logrus.ErrorLevel)
	default:
		logrus.SetLevel(logrus.InfoLevel)
	}
}

func setupRouter(cfg *config.Config) *gin.Engine {
	router := gin.New()

	// 添加中间件
	router.Use(middleware.Logger())
	router.Use(middleware.Recovery())
	router.Use(middleware.CORS())
	router.Use(middleware.RateLimit(cfg.RateLimit))
	router.Use(middleware.Metrics())

	// 健康检查
	router.GET("/health", api.HealthCheck)
	router.GET("/ready", api.ReadinessCheck)

	// Prometheus指标
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Swagger文档
	router.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// WebSocket端点
	router.GET("/ws", websocket.HandleWebSocket)

	// API路由组
	v1 := router.Group("/api/v1")
	{
		// VPN检测相关API
		detection := v1.Group("/detection")
		{
			detection.POST("/analyze", api.AnalyzeTraffic)
			detection.GET("/results", api.GetDetectionResults)
			detection.GET("/results/:id", api.GetDetectionResult)
			detection.POST("/batch", api.BatchAnalyze)
		}

		// 系统监控API
		monitoring := v1.Group("/monitoring")
		{
			monitoring.GET("/stats", api.GetSystemStats)
			monitoring.GET("/performance", api.GetPerformanceMetrics)
			monitoring.GET("/alerts", api.GetAlerts)
			monitoring.POST("/alerts/:id/ack", api.AcknowledgeAlert)
		}

		// 配置管理API
		config := v1.Group("/config")
		{
			config.GET("/", api.GetConfig)
			config.PUT("/", api.UpdateConfig)
			config.GET("/baselines", api.GetBaselines)
			config.PUT("/baselines", api.UpdateBaselines)
		}

		// 模型管理API
		models := v1.Group("/models")
		{
			models.GET("/", api.ListModels)
			models.GET("/:id", api.GetModel)
			models.POST("/", api.UploadModel)
			models.PUT("/:id/activate", api.ActivateModel)
			models.DELETE("/:id", api.DeleteModel)
		}

		// 数据管理API
		data := v1.Group("/data")
		{
			data.GET("/flows", api.GetFlows)
			data.GET("/packets", api.GetPackets)
			data.POST("/export", api.ExportData)
			data.DELETE("/cleanup", api.CleanupData)
		}
	}

	return router
}

func gracefulShutdown(server *http.Server) {
	// 等待中断信号
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logrus.Info("开始优雅关闭服务器...")

	// 设置超时上下文
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// 关闭HTTP服务器
	if err := server.Shutdown(ctx); err != nil {
		logrus.Errorf("服务器关闭错误: %v", err)
	}

	logrus.Info("服务器已成功关闭")
}