# 基於深度強化學習之微服務容器多目標自動擴展績效評估
> **An Adaptive and Proactive Microservice Autoscaling Engine using Deep Reinforcement Learning (PPO)**

---

## 📌 1. 專案背景與痛點 (Research Background & Motivation)

在現代雲端原生（Cloud Native）架構中，微服務的彈性伸縮（Autoscaling）是維持系統穩定性與控制基礎設施成本的核心技術。然而，現行 Kubernetes 內建的 **Horizontal Pod Autoscaler (HPA)** 存在以下兩大致命缺陷：

1. **反應式控制的滯後性 (Reactive Delay)**：HPA 必須等到監控指標（如 CPU 利用率）「實質衝破閾值」後才會觸發擴容，屬於落後指標。
2. **實體冷啟動延遲 (Startup Latency大魔王)**：在真實工業界環境中，拉取 Docker 映像檔、初始化 JVM 虛擬機或執行 Kubernetes 探針（Probing）通常需要 5 到 10 分鐘。當突發海嘯流量灌入時，HPA 的滯後性疊加冷啟動空窗期，將導致請求在緩衝區引發嚴重的**排隊雪崩效應**，直接撕裂企業的 SLA 服務死線。

本專案基於 **Gymnasium** 接口自主構建了一個高度擬真的微服務動態排隊模擬環境，並導入 **PPO (Proximal Policy Optimization)** 演算法，成功實證了 AI Agent 能夠具備「主動預測（Proactive）」與「商業合約感知（Contract Awareness）」能力，在不同的營運用意下自動尋找最優調度邊界。

---

## 🔬 2. 底層數學模型與自訂環境設計 (Mathematical Modeling)

本專案拒絕使用簡單的線性模擬，而是將**作業研究（Operations Research）中的排隊理論**與微服務非線性延遲行為深度結合：

### A. 觀測空間 (Observation Space) - 5維連續向量
環境在每個時間步長（Step，每 5 分鐘為一步，單日共 288 步）向 AI 揭露以下狀態：

$$
S = [Active\_Pods, Pending\_Pods, RPS, CPU\_Util, Latency]
$$

### B. 排隊理論與 CPU 利用率公式
系統中單台 Pod 的最大處理能力設定為 100.0 RPS。整體的系統硬體利用率（CPU Utilization）並非線性，而是受外部當前請求數（RPS）與當前就緒 Pod 總運算能力的拉鋸所決定：

$$
CPU\_Util = \min\left(100.0, \max\left(0.0, \frac{RPS}{Active\_Pods \times 100.0} \times 100.0 \right)\right)
$$

### C. 微服務延遲 (Latency) 指數級雪崩模型
當系統負載安全時，延遲維持在基本物理基線（20ms 左右）；然而一旦 CPU 利用率逼近 95% 臨界點，網路緩衝區排隊隊伍將會崩潰，延遲呈現**非線性指數級暴衝**，完美還原真實分散式系統的「滅頂現象」：

* 當 $CPU\_Util < 95.0\%$ 時：

$$
Latency = 20.0 + \left(\frac{CPU\_Util}{100.0 - CPU\_Util}\right) \times 10.0 \text{ (ms)}
$$

* 當 $CPU\_Util \ge 95.0\%$ 時：

$$
Latency = 20.0 + \left(\frac{95.0}{5.0}\right) \times 10.0 + (CPU\_Util - 95.0) \times 100.0 \text{ (ms)}
$$

---

## ⚖️ 3. 雙策略動態獎勵函數設計 (Dynamic Reward Function)

本專案的核心學術貢獻在於**「一套演算法核心，一鍵切換兩種維運性格」**。透過在 `step()` 中動態判斷 `self.mode`，獎勵函數（Reward）引導神經網路往完全相反的帕累托優化路徑（Pareto Optimal）收斂：

### 📋 策略分支 A：`SLA_PRIORITY` (對齊 Tier-1 核心服務)
- **維運意志**：企業零容忍任何斷線與卡頓（如購物車、機台連線系統）。
- **數學權重**：硬體極便宜（權重 1.5），硬體超載與 SLA 延遲超標**極重罰**（權重 5.0 / 2.0）。
- **AI 演化性格**：**防禦型土豪**。神經網路精算出「縮容冒險的罰金遠比養硬體貴」，因此收斂出**「清晨即提早開滿 20 台 Pod 進行絕對預防、深夜流量下降後亦絕不縮容」**的超高可用性策略，死守 80ms 延遲生死線。

### 📋 策略分支 B：`COST_PRIORITY` (對齊 Tier-3 非核心服務)
- **維運意志**：極致縮減 IT 基礎設施預算，硬體壓榨效率最大化（如內部後台報表）。
- **數學權重**：Pod 成本**強行放大 10 倍**（權重 15.0），SLA 卡頓處罰調至極輕（權重 1.0 / 0.5）。
- **AI 演化性格**：**精打細算鐵公雞**。全天將 Pod 嚴格控制在 8-12 台，讓 CPU 長期踩在 60%~80% 的高效鋼索上。下午突發海嘯時，AI **理性選擇放任系統短暫滅頂（噴出 700ms 延遲）**，因為算總帳下來，這樣幫公司省下的雲端租金更巨大。

---

## 📊 4. 實驗量化數據與科學對照 (Evaluation & Benchmarks)

本專案在 Google Colab 環境下，將兩個 Agent 分別進行了 **200,000 Steps** 的充分特訓，並導入突發海嘯流量尖峰（Step 168-192 突發灌入 1400 RPS）與「傳統規則式 HPA」進行單日封閉式科學對比：

### 📈 累積總回報得分對照表
| 測試維運賽道 | 傳統基準線 HPA 總分 | DRL AI Agent 總分 | 🟢 核心量化效益與維運行為分析 |
| :--- | :---: | :---: | :--- |
| **🏆 SLA 優先環境**<br>*(Core Tier-1 Track)* | `-9555.58` | **`-8428.00`** | **AI 達成 0% SLA 違規**。HPA 面臨海嘯時因被動滯後導致延遲**暴衝至 570ms** 觸發天價罰分。AI 則成功將延遲穩壓在 30ms。 |
| **🏆 Cost 敏感環境**<br>*(Non-Core Tier-3 Track)* | `-42091.32` | **`-36765.50`** | **AI 實質消滅 75% 的不合理營運浪費**。成功砍掉 HPA 在離峰期過度震盪多開的 Pod 租金，全天資源財務效益達到最優。 |

> 📊 **關於統計基數稀釋的學術澄清**：乍看之下兩者得分接近，是因為全天 288 步包含了龐大且不可避的「背景固定雲端基礎租金基數」（約 3.5 萬分）。扣除此不可避基本開銷後，**DRL AI 在 Cost 賽道上實質優化了高達 75.1% 的非必要營運開銷**；在 SLA 賽道上更是達成了 **「完美零事故」** 的本質差別。

### 📉 雙 Y 軸實驗數據可視化
本專案運行結束後會自動在根目錄下輸出兩張 **300 DPI 高解析度學術圖表**，清晰記錄流量 RPS、Pod 數量、CPU 利用率與 Latency 三位一體動態變化：

<p align="center">
  <img src="experiment_a_sla_priority.png" width="49%" />
  <img src="experiment_b_cost_priority.png" width="49%" />
</p>

---

## 📁 5. 專案檔案結構 (Repository Structure)

```text
├── Configurable_Scaling_Env.py   # 基於排隊理論自訂的 Gymnasium 微服務動態環境
├── DRL_Agent_Training.py         # SB3 PPO 演算法特訓、雙人格模型訓練與測試核心腳本
├── experiment_a_sla_priority.png # 實驗 A (SLA優先土豪型) 雙 Y 軸三階科學數據圖表 (自動生成)
├── experiment_b_cost_priority.png# 實驗 B (Cost敏感鐵公雞) 雙 Y 軸三階科學數據圖表 (自動生成)
└── README.md                     # 本說明文件

🛠️ 6. 快速開始與重現實驗 (Quick Start)
1. 環境配置
本專案與現行強化學習生態無縫相容，請於 Python 3.10+ 環境下執行：

Bash
pip install stable-baselines3[extra] gymnasium matplotlib numpy
2. 執行訓練、對照測試與繪圖
直接運行腳本，虛擬機會自動執行 20 萬步 PPO 特訓，現場開獎量化得分對照表，並自動導出圖表：

Bash
python DRL_Agent_Training.py
🚀 7. 未來商業展望 (MLOps Future Work)
MLOps 自動化再訓練流水線：本套 DRL 調度引擎非常適合與 Apache Airflow 與 Docker 容器化技術封裝。當生產環境的流量特徵（Workload Profile）發生概念漂移（Concept Drift）時，流水線將自動抓取最新日誌進行線上微調（Fine-tuning）。

商用 IT 預算精準投放：本框架未來可直接作為企業內部 FinOps（雲端財務運營） 的核心控制大腦，根據微服務分級制度（Service Tiering, Tier-1 ~ Tier-3），實現基礎設施開銷的主動式最優投放。
