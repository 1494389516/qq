package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// Go并发反击执行器
// 基于风控算法专家的理论框架，实现高并发DDoS反击策略
// 核心优势：goroutine轻量级并发 + 高效网络处理 + 自动负载均衡

// AttackSource 攻击源信息
type AttackSource struct {
	IP          string    `json:"ip"`
	Port        int       `json:"port"`
	AttackType  string    `json:"attack_type"`
	RiskScore   float64   `json:"risk_score"`
	Confidence  float64   `json:"confidence"`
	FirstSeen   time.Time `json:"first_seen"`
	LastSeen    time.Time `json:"last_seen"`
	PacketCount int64     `json:"packet_count"`
}

// CounterMeasure 反击措施
type CounterMeasure struct {
	Type        string    `json:"type"`        // "rate_limit", "block", "divert"
	Target      string    `json:"target"`      // 目标IP或网段
	Duration    int       `json:"duration"`    // 持续时间(秒)
	Intensity   float64   `json:"intensity"`   // 反击强度[0.0-1.0]
	ExecutedAt  time.Time `json:"executed_at"`
	Status      string    `json:"status"`      // "pending", "active", "completed", "failed"
}

// ConcurrentCounterAttacker 并发反击执行器
type ConcurrentCounterAttacker struct {
	// 工作池配置
	maxWorkers   int
	taskQueue    chan AttackTask
	workerPool   chan chan AttackTask
	workers      []*Worker
	
	// 反击策略配置
	rateLimit    map[string]int     // IP -> 限制速率
	blockedIPs   map[string]time.Time // IP -> 封禁到期时间
	activeTasks  map[string]*CounterMeasure // 活跃任务
	
	// 同步控制
	mutex        sync.RWMutex
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
	
	// 性能监控
	metrics      *PerformanceMetrics
}

// AttackTask 攻击任务
type AttackTask struct {
	ID          string         `json:"id"`
	Source      AttackSource   `json:"source"`
	Measure     CounterMeasure `json:"measure"`
	CreatedAt   time.Time      `json:"created_at"`
	Priority    int            `json:"priority"`  // 优先级 1-10
}

// Worker 工作协程
type Worker struct {
	ID         int
	taskChan   chan AttackTask
	workerPool chan chan AttackTask
	attacker   *ConcurrentCounterAttacker
	quit       chan bool
}

// PerformanceMetrics 性能指标
type PerformanceMetrics struct {
	TotalTasks      int64     `json:"total_tasks"`
	CompletedTasks  int64     `json:"completed_tasks"`
	FailedTasks     int64     `json:"failed_tasks"`
	ActiveWorkers   int       `json:"active_workers"`
	QueueLength     int       `json:"queue_length"`
	AvgResponseTime float64   `json:"avg_response_time_ms"`
	Throughput      float64   `json:"throughput_per_second"`
	LastUpdated     time.Time `json:"last_updated"`
	
	mutex sync.RWMutex
}

// NewConcurrentCounterAttacker 创建并发反击执行器
func NewConcurrentCounterAttacker(maxWorkers int) *ConcurrentCounterAttacker {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &ConcurrentCounterAttacker{
		maxWorkers:  maxWorkers,
		taskQueue:   make(chan AttackTask, maxWorkers*10), // 缓冲队列
		workerPool:  make(chan chan AttackTask, maxWorkers),
		rateLimit:   make(map[string]int),
		blockedIPs:  make(map[string]time.Time),
		activeTasks: make(map[string]*CounterMeasure),
		ctx:         ctx,
		cancel:      cancel,
		metrics:     &PerformanceMetrics{},
	}
}

// Start 启动反击系统
func (c *ConcurrentCounterAttacker) Start() error {
	log.Printf("🚀 启动并发反击执行器，工作协程数: %d", c.maxWorkers)
	
	// 启动工作协程池
	c.workers = make([]*Worker, c.maxWorkers)
	for i := 0; i < c.maxWorkers; i++ {
		worker := &Worker{
			ID:         i,
			taskChan:   make(chan AttackTask),
			workerPool: c.workerPool,
			attacker:   c,
			quit:       make(chan bool),
		}
		c.workers[i] = worker
		c.wg.Add(1)
		go worker.Start()
	}
	
	// 启动任务分发器
	c.wg.Add(1)
	go c.dispatcher()
	
	// 启动性能监控
	c.wg.Add(1)
	go c.performanceMonitor()
	
	// 启动清理协程
	c.wg.Add(1)
	go c.cleanupExpiredBlocks()
	
	return nil
}

// Stop 停止反击系统
func (c *ConcurrentCounterAttacker) Stop() {
	log.Println("🛑 停止并发反击执行器...")
	
	c.cancel()
	
	// 停止所有worker
	for _, worker := range c.workers {
		worker.Stop()
	}
	
	c.wg.Wait()
	log.Println("✅ 并发反击执行器已停止")
}

// ExecuteCounterMeasures 执行反击措施
func (c *ConcurrentCounterAttacker) ExecuteCounterMeasures(sources []AttackSource) error {
	for _, source := range sources {
		// 生成反击策略
		measure := c.generateCounterMeasure(source)
		
		task := AttackTask{
			ID:        fmt.Sprintf("task_%d_%s", time.Now().Unix(), source.IP),
			Source:    source,
			Measure:   measure,
			CreatedAt: time.Now(),
			Priority:  c.calculatePriority(source),
		}
		
		// 提交到任务队列（非阻塞）
		select {
		case c.taskQueue <- task:
			c.metrics.mutex.Lock()
			c.metrics.TotalTasks++
			c.metrics.mutex.Unlock()
		case <-time.After(100 * time.Millisecond):
			log.Printf("⚠️ 任务队列已满，丢弃任务: %s", source.IP)
			c.metrics.mutex.Lock()
			c.metrics.FailedTasks++
			c.metrics.mutex.Unlock()
		}
	}
	return nil
}

// generateCounterMeasure 生成反击策略
func (c *ConcurrentCounterAttacker) generateCounterMeasure(source AttackSource) CounterMeasure {
	measure := CounterMeasure{
		Target:     source.IP,
		ExecutedAt: time.Now(),
		Status:     "pending",
	}
	
	// 基于风险评分和攻击类型确定反击策略
	switch {
	case source.RiskScore >= 0.9:
		// 高风险：立即封禁
		measure.Type = "block"
		measure.Duration = 3600 // 1小时
		measure.Intensity = 1.0
		
	case source.RiskScore >= 0.7:
		// 中等风险：限流
		measure.Type = "rate_limit"
		measure.Duration = 1800 // 30分钟
		measure.Intensity = 0.3  // 限制到30%
		
	case source.RiskScore >= 0.5:
		// 低风险：流量牵引
		measure.Type = "divert"
		measure.Duration = 900  // 15分钟
		measure.Intensity = 0.1
		
	default:
		// 观察模式
		measure.Type = "monitor"
		measure.Duration = 300
		measure.Intensity = 0.0
	}
	
	return measure
}

// calculatePriority 计算任务优先级
func (c *ConcurrentCounterAttacker) calculatePriority(source AttackSource) int {
	priority := 5 // 默认优先级
	
	// 基于风险评分调整优先级
	if source.RiskScore >= 0.9 {
		priority = 10 // 最高优先级
	} else if source.RiskScore >= 0.7 {
		priority = 8
	} else if source.RiskScore >= 0.5 {
		priority = 6
	}
	
	// 基于攻击类型调整
	switch source.AttackType {
	case "ddos":
		priority += 2
	case "brute_force":
		priority += 1
	}
	
	return min(priority, 10)
}

// dispatcher 任务分发器
func (c *ConcurrentCounterAttacker) dispatcher() {
	defer c.wg.Done()
	
	for {
		select {
		case task := <-c.taskQueue:
			// 获取可用worker
			select {
			case workerTaskChan := <-c.workerPool:
				// 分发任务给worker
				select {
				case workerTaskChan <- task:
				case <-c.ctx.Done():
					return
				}
			case <-c.ctx.Done():
				return
			}
		case <-c.ctx.Done():
			return
		}
	}
}

// Worker.Start 启动工作协程
func (w *Worker) Start() {
	defer w.attacker.wg.Done()
	
	for {
		// 将worker通道放入池中
		w.workerPool <- w.taskChan
		
		select {
		case task := <-w.taskChan:
			// 执行反击任务
			start := time.Now()
			err := w.executeTask(task)
			duration := time.Since(start)
			
			// 更新指标
			w.attacker.metrics.mutex.Lock()
			if err != nil {
				w.attacker.metrics.FailedTasks++
				log.Printf("❌ Worker %d 执行任务失败: %s, 错误: %v", w.ID, task.ID, err)
			} else {
				w.attacker.metrics.CompletedTasks++
				log.Printf("✅ Worker %d 完成任务: %s, 耗时: %v", w.ID, task.ID, duration)
			}
			
			// 更新平均响应时间
			totalTasks := w.attacker.metrics.CompletedTasks + w.attacker.metrics.FailedTasks
			if totalTasks > 0 {
				w.attacker.metrics.AvgResponseTime = 
					(w.attacker.metrics.AvgResponseTime*float64(totalTasks-1) + duration.Seconds()*1000) / float64(totalTasks)
			}
			w.attacker.metrics.mutex.Unlock()
			
		case <-w.quit:
			return
		case <-w.attacker.ctx.Done():
			return
		}
	}
}

// Worker.Stop 停止工作协程
func (w *Worker) Stop() {
	w.quit <- true
}

// executeTask 执行具体的反击任务
func (w *Worker) executeTask(task AttackTask) error {
	log.Printf("🎯 Worker %d 执行反击: %s -> %s (强度: %.2f)", 
		w.ID, task.Measure.Type, task.Source.IP, task.Measure.Intensity)
	
	// 根据反击类型执行相应操作
	switch task.Measure.Type {
	case "block":
		return w.executeBlock(task)
	case "rate_limit":
		return w.executeRateLimit(task)
	case "divert":
		return w.executeDivert(task)
	case "monitor":
		return w.executeMonitor(task)
	default:
		return fmt.Errorf("未知的反击类型: %s", task.Measure.Type)
	}
}

// executeBlock 执行IP封禁
func (w *Worker) executeBlock(task AttackTask) error {
	w.attacker.mutex.Lock()
	defer w.attacker.mutex.Unlock()
	
	// 添加到封禁列表
	expireTime := time.Now().Add(time.Duration(task.Measure.Duration) * time.Second)
	w.attacker.blockedIPs[task.Source.IP] = expireTime
	
	// 更新任务状态
	task.Measure.Status = "active"
	w.attacker.activeTasks[task.ID] = &task.Measure
	
	log.Printf("🚫 封禁IP: %s, 到期时间: %s", task.Source.IP, expireTime.Format("15:04:05"))
	
	// 模拟与防火墙/网络设备的API交互
	time.Sleep(10 * time.Millisecond)
	
	return nil
}

// executeRateLimit 执行限流
func (w *Worker) executeRateLimit(task AttackTask) error {
	w.attacker.mutex.Lock()
	defer w.attacker.mutex.Unlock()
	
	// 计算限流速率
	baseRate := 1000 // 基础速率 1000 pps
	limitedRate := int(float64(baseRate) * task.Measure.Intensity)
	w.attacker.rateLimit[task.Source.IP] = limitedRate
	
	// 更新任务状态
	task.Measure.Status = "active"
	w.attacker.activeTasks[task.ID] = &task.Measure
	
	log.Printf("🚦 限流IP: %s, 限制速率: %d pps", task.Source.IP, limitedRate)
	
	// 模拟限流配置
	time.Sleep(5 * time.Millisecond)
	
	return nil
}

// executeDivert 执行流量牵引
func (w *Worker) executeDivert(task AttackTask) error {
	log.Printf("🔀 流量牵引: %s, 强度: %.2f", task.Source.IP, task.Measure.Intensity)
	
	// 模拟流量牵引到蜜罐或清洗设备
	time.Sleep(15 * time.Millisecond)
	
	return nil
}

// executeMonitor 执行监控
func (w *Worker) executeMonitor(task AttackTask) error {
	log.Printf("👁️ 监控IP: %s", task.Source.IP)
	
	// 加强对该IP的监控
	time.Sleep(2 * time.Millisecond)
	
	return nil
}

// performanceMonitor 性能监控协程
func (c *ConcurrentCounterAttacker) performanceMonitor() {
	defer c.wg.Done()
	
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	var lastCompleted int64
	
	for {
		select {
		case <-ticker.C:
			c.metrics.mutex.Lock()
			
			// 计算吞吐量
			currentCompleted := c.metrics.CompletedTasks
			c.metrics.Throughput = float64(currentCompleted - lastCompleted)
			lastCompleted = currentCompleted
			
			// 更新其他指标
			c.metrics.QueueLength = len(c.taskQueue)
			c.metrics.ActiveWorkers = c.maxWorkers
			c.metrics.LastUpdated = time.Now()
			
			c.metrics.mutex.Unlock()
			
		case <-c.ctx.Done():
			return
		}
	}
}

// cleanupExpiredBlocks 清理过期封禁
func (c *ConcurrentCounterAttacker) cleanupExpiredBlocks() {
	defer c.wg.Done()
	
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			c.mutex.Lock()
			now := time.Now()
			for ip, expireTime := range c.blockedIPs {
				if now.After(expireTime) {
					delete(c.blockedIPs, ip)
					log.Printf("🔓 解封IP: %s", ip)
				}
			}
			c.mutex.Unlock()
			
		case <-c.ctx.Done():
			return
		}
	}
}

// GetMetrics 获取性能指标
func (c *ConcurrentCounterAttacker) GetMetrics() *PerformanceMetrics {
	c.metrics.mutex.RLock()
	defer c.metrics.mutex.RUnlock()
	
	// 返回指标副本
	return &PerformanceMetrics{
		TotalTasks:      c.metrics.TotalTasks,
		CompletedTasks:  c.metrics.CompletedTasks,
		FailedTasks:     c.metrics.FailedTasks,
		ActiveWorkers:   c.metrics.ActiveWorkers,
		QueueLength:     c.metrics.QueueLength,
		AvgResponseTime: c.metrics.AvgResponseTime,
		Throughput:      c.metrics.Throughput,
		LastUpdated:     c.metrics.LastUpdated,
	}
}

// REST API 服务器
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // 允许跨域
	},
}

// setupRoutes 设置API路由
func setupRoutes(attacker *ConcurrentCounterAttacker) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	
	// 健康检查
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"timestamp": time.Now(),
		})
	})
	
	// 获取性能指标
	r.GET("/api/v1/metrics", func(c *gin.Context) {
		metrics := attacker.GetMetrics()
		c.JSON(http.StatusOK, metrics)
	})
	
	// 提交反击任务
	r.POST("/api/v1/counter-attack", func(c *gin.Context) {
		var sources []AttackSource
		if err := c.ShouldBindJSON(&sources); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		
		err := attacker.ExecuteCounterMeasures(sources)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(http.StatusOK, gin.H{
			"message": "反击任务已提交",
			"count": len(sources),
		})
	})
	
	// WebSocket实时指标推送
	r.GET("/ws/metrics", func(c *gin.Context) {
		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			log.Printf("WebSocket升级失败: %v", err)
			return
		}
		defer conn.Close()
		
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				metrics := attacker.GetMetrics()
				if err := conn.WriteJSON(metrics); err != nil {
					log.Printf("WebSocket写入失败: %v", err)
					return
				}
			}
		}
	})
	
	return r
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	// 创建并发反击执行器
	attacker := NewConcurrentCounterAttacker(100) // 100个工作协程
	
	// 启动反击系统
	if err := attacker.Start(); err != nil {
		log.Fatalf("启动反击系统失败: %v", err)
	}
	
	// 设置API服务器
	router := setupRoutes(attacker)
	
	// 启动HTTP服务器
	log.Println("🚀 Go并发反击执行器启动在端口 8002")
	log.Fatal(http.ListenAndServe(":8002", router))
}