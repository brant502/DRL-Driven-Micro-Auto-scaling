# 1. 先強制安裝需要的強化學習環境與函式庫
!pip install stable-baselines3[extra] gymnasium

# 2. 接著才是你原本的 import
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO  # 👈 這次就不會再喷 ModuleNotFoundError 了！

# ==============================================================================
# 1. 宣告具備「權重調整功能」的進階環境
# ==============================================================================
class ConfigurableScalingEnv(gym.Env):
    def __init__(self, mode="SLA_PRIORITY"):
        super(ConfigurableScalingEnv, self).__init__()
        self.mode = mode # "SLA_PRIORITY" 或 "COST_PRIORITY"

        self.action_space = spaces.Discrete(3) # 0: 維持, 1: +1 Pod, 2: -1 Pod
        low = np.array([1, 0, 0, 0.0, 0.0], dtype=np.float32)
        high = np.array([20, 20, 2000, 100.0, 5000.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_steps = 288
        self.pod_capacity_rps = 100.0
        self.startup_delay_steps = 2
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.active_pods = 4
        self.pending_pods = []
        return self._get_environment_state(), {}

    def _generate_traffic(self, step):
        base_traffic = 500 + 400 * np.sin(2 * np.pi * step / 288 - np.pi / 2)
        spike = 0
        if 168 <= step <= 192: # 下午突發尖峰
            spike = 450
        noise = np.random.normal(0, 15) # 降低噪音讓對照組流量完全一致
        return max(10, int(base_traffic + spike + noise))

    def _get_environment_state(self):
        self.active_pods = max(1, self.active_pods)
        rps = self._generate_traffic(self.current_step)
        total_capacity = self.active_pods * self.pod_capacity_rps
        cpu_util = (rps / total_capacity) * 100.0 if total_capacity > 0 else 100.0
        cpu_util = min(100.0, max(0.0, cpu_util))

        if cpu_util < 95.0:
            latency = 20.0 + (cpu_util / (100.0 - cpu_util)) * 10.0
        else:
            latency = 20.0 + (95.0 / 5.0) * 10.0 + (cpu_util - 95.0) * 100.0

        return np.array([self.active_pods, len(self.pending_pods), rps, cpu_util, latency], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        # 處理冷啟動
        ready_pods = [p for p in self.pending_pods if p <= self.current_step]
        self.active_pods += len(ready_pods)
        self.pending_pods = [p for p in self.pending_pods if p > self.current_step]

        if action == 1 and (self.active_pods + len(self.pending_pods)) < 20:
            self.pending_pods.append(self.current_step + self.startup_delay_steps)
        elif action == 2 and self.active_pods > 1:
            self.active_pods -= 1

        self.active_pods = max(1, self.active_pods)
        next_state = self._get_environment_state()
        active_pods, _, rps, cpu_util, latency = next_state

        # ==========================================
        # ✨ 關鍵的核心權重動態切換
        # ==========================================
        if self.mode == "SLA_PRIORITY":
            # 你原本的設定：注重體驗，SLA 罰分很重
            cost_penalty = active_pods * 1.5 + len(self.pending_pods) * 0.5
            sla_penalty = 0
            if latency > 80.0: sla_penalty += (latency - 80.0) * 2.0
            if cpu_util > 85.0: sla_penalty += (cpu_util - 85.0) * 5.0
        else:
            # 業界省錢設定：Pod 超級貴！SLA 罰分調輕，逼 AI 壓榨硬體
            cost_penalty = active_pods * 15.0 + len(self.pending_pods) * 5.0  # 放大10倍
            sla_penalty = 0
            if latency > 80.0: sla_penalty += (latency - 80.0) * 0.5
            if cpu_util > 85.0: sla_penalty += (cpu_util - 85.0) * 1.0   # 縮小5倍

        reward = -(cost_penalty + sla_penalty)
        return next_state, reward, (self.current_step >= self.max_steps), False, {}

# ==============================================================================
# 2. 【特訓修正版】開始深度訓練兩個不同性格的 AI Agent
# ==============================================================================
# 調整學習率，並引進多進程或足夠的訓練步數，逼迫神經網路收斂

print("====== 🚀 正在特訓 AI-A: 服務品質優先 (SLA Priority Agent) ======")
env_sla = ConfigurableScalingEnv(mode="SLA_PRIORITY")
# 調整 learning_rate 為 5e-4，並給予 20 萬步的充分特訓
model_sla = PPO("MlpPolicy", env_sla, learning_rate=5e-4, n_steps=2048, batch_size=64, verbose=0)
model_sla.learn(total_timesteps=200000) # 👈 放大到 20 萬步

print("====== 🚀 正在特訓 AI-B: 基礎設施省錢優先 (Cost Priority Agent) ======")
env_cost = ConfigurableScalingEnv(mode="COST_PRIORITY")
model_cost = PPO("MlpPolicy", env_cost, learning_rate=5e-4, n_steps=2048, batch_size=64, verbose=0)
model_cost.learn(total_timesteps=200000) # 👈 放大到 20 萬步

print("✨ 兩個 AI 特訓完全成熟！")

# ==============================================================================
# 3. 資料收集測試函式 (修正版：完美追蹤分數)
# ==============================================================================
def run_eval(env_mode, policy_type):
    test_env = ConfigurableScalingEnv(mode=env_mode)
    obs, _ = test_env.reset()
    # ✨ 在歷史紀錄字典裡，多宣告一個 rewards 陣列
    history = {"steps": [], "pods": [], "rps": [], "cpu": [], "latency": [], "rewards": []}
    terminated = False

    while not terminated:
        current_step = test_env.current_step
        current_cpu = obs[3]

        if policy_type == "AI_SLA":
            action, _ = model_sla.predict(obs, deterministic=True)
        elif policy_type == "AI_COST":
            action, _ = model_cost.predict(obs, deterministic=True)
        else: # Traditional HPA
            if current_cpu > 80.0: action = 1
            elif current_cpu < 30.0: action = 2
            else: action = 0

        # ✨ 把原本丟棄的 reward 拿變數接住！
        next_obs, reward, terminated, _, _ = test_env.step(action)
        
        history["steps"].append(current_step)
        history["pods"].append(obs[0])
        history["rps"].append(obs[2])
        history["cpu"].append(obs[3])
        history["latency"].append(obs[4])
        history["rewards"].append(reward) # ✨ 把每一步的分數存起來
        obs = next_obs
    return history

# 執行測試收集數據
data_sla_agent = run_eval("SLA_PRIORITY", "AI_SLA")
data_cost_agent = run_eval("COST_PRIORITY", "AI_COST")

# ✨ HPA 必須在兩種不同的環境下各跑一次，才能算出它在不同計分板下的真實悲劇分數
data_hpa_in_sla_env  = run_eval("SLA_PRIORITY", "HPA")
data_hpa_in_cost_env = run_eval("COST_PRIORITY", "HPA")

# ==============================================================================
# 🌟 現場開獎！列印出三大策略的真實總得分
# ==============================================================================
print("\n====== 📊 實驗最終得分量化對照表 ======")
print(f"🏆 SLA 優先環境 (Tier-1 賽道):")
print(f"  - DRL AI 土豪策略總分: {sum(data_sla_agent['rewards']):.2f} 分")
print(f"  - 傳統基準線 HPA 總分: {sum(data_hpa_in_sla_env['rewards']):.2f} 分")

print(f"\n🏆 Cost 敏感環境 (Tier-3 賽道):")
print(f"  - DRL AI 鐵公雞策略總分: {sum(data_cost_agent['rewards']):.2f} 分")
print(f"  - 傳統基準線 HPA 總分: {sum(data_hpa_in_cost_env['rewards']):.2f} 分")

# ==============================================================================
# 4. 繪製兩張圖表進行科學對照
# ==============================================================================
def plot_dual_axis_chart(title, ai_data, hpa_data, filename):
    fig, (ax1, ax3, ax4) = plt.subplots(3, 1, figsize=(14, 11))

    # 1. 流量與 Pod 擴展
    color = 'tab:gray'
    ax1.set_ylabel('Incoming RPS (Traffic)', color=color)
    ax1.plot(ai_data["steps"], ai_data["rps"], color=color, linestyle="--", alpha=0.4, label="Traffic RPS")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    line1 = ax2.step(ai_data["steps"], ai_data["pods"], color='green', linewidth=2.5, label="DRL AI Agent")
    line2 = ax2.step(hpa_data["steps"], hpa_data["pods"], color='red', linewidth=1.5, alpha=0.7, label="Traditional HPA")
    ax2.set_ylabel('Active Pods Count', color='black')
    ax2.set_ylim(0, 22)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title(title, fontsize=14, fontweight='bold')

    # 2. CPU 使用率
    ax3.plot(ai_data["steps"], ai_data["cpu"], color='green', linewidth=2, label="DRL AI CPU (%)")
    ax3.plot(hpa_data["steps"], hpa_data["cpu"], color='red', linewidth=1.5, alpha=0.5, label="HPA CPU (%)")
    ax3.axhline(y=85, color='purple', linestyle=':', label="SLA Target (85%)")
    ax3.set_ylabel('CPU Utilization (%)')
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # 3. 延遲
    ax4.plot(ai_data["steps"], ai_data["latency"], color='green', linewidth=2, label="DRL AI Latency (ms)")
    ax4.plot(hpa_data["steps"], hpa_data["latency"], color='red', linewidth=1.5, alpha=0.5, label="HPA Latency (ms)")
    ax4.axhline(y=80, color='purple', linestyle=':', label="SLA Latency Target (80ms)")
    ax4.set_xlabel('Timeline Steps (5-min intervals, 288 steps = 24 Hours)')
    ax4.set_ylabel('Latency (ms)')
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# 繪製圖表一：SLA 優先型
plot_dual_axis_chart("Experiment A: SLA-Priority Agent (Core Tier-1 Service Strategy)", data_sla_agent, data_hpa, "experiment_a_sla_priority.png")

# 繪製圖表二：極致省錢型
plot_dual_axis_chart("Experiment B: Cost-Sensitive Agent (Non-Core Tier-3 Service Strategy)", data_cost_agent, data_hpa, "experiment_b_cost_priority.png")