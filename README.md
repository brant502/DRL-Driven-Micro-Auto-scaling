# 基於深度強化學習之微服務容器多目標自動擴展績效評估
> **An Adaptive Microservice Autoscaling Engine using Deep Reinforcement Learning (PPO)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)

本專案構建了一個基於 **深度強化學習 (DRL)** 的雲端原生微服務主動式（Proactive）自動擴展機制。傳統的 Kubernetes HPA（Horizontal Pod Autoscaler）依賴被動的閾值觸發，面對突發流量尖峰時常因**容器冷啟動延遲（Startup Latency）**導致服務層次協定（SLA）崩潰。

本專案透過自訂 **Gymnasium** 環境模擬微服務排隊理論模型，並導入 **PPO (Proximal Policy Optimization)** 演算法，實證了 AI Agent 能夠在「死守 SLA 品質」與「極致壓榨雲端租金成本」兩種極端商業意志下，自主演化出最適的調度邊界（Pareto Optimal）。

---

## 🌟 核心架構與技術亮點

- **環境架構（Custom Gymnasium Env）**：內建自訂 5 維觀測空間（Active Pods, Pending Pods, RPS, CPU, Latency），底層非線性公式基於**排隊理論（Queueing Theory）**，真實還原流量與 CPU 利用率的拉鋸關係。
- **物理制約模擬**：嚴格實作 `startup_delay_steps = 2` 的冷啟動空窗期，逼迫 AI 必須具備時序預測與主動防禦能力。
- **雙商業人格權重切換**：透過調整獎勵函數（Reward Function）的懲罰係數，實現「一套演算法框架，一鍵訓練兩種專家選手」：
  - `SLA_PRIORITY`：對齊 Tier-1 核心服務（零容忍卡頓，高可用性防禦）。
  - `COST_PRIORITY`：對齊 Tier-3 非核心服務（成本極度敏感，硬體高效壓榨）。

---

## 📊 實驗最終得分與量化績效

本專案在 Google Colab 環境下對兩個 AI Agent 進行了 **200,000 Steps** 的深度特訓，並以單日複合流量（正弦波常態 + Step 170 突發 1400 RPS 海嘯尖峰）與「傳統 HPA 基準線」進行封閉式科學對照：

| 評估賽道 \ 維運策略 | 傳統基準線 HPA 總分 | DRL AI Agent 總分 | 真實維運物理現象與決策邏輯 |
| :--- | :---: | :---: | :--- |
| **🏆 SLA 優先環境**<br>*(Core Tier-1 Service)* | `-9555.58` | **`-8428.00`**<br>`(績效提升)` | **AI 達成 0% SLA 違規**，延遲穩壓在 30ms。AI 展現「風險規避」特質，清晨即主動提早開滿 20 台 Pod 進行防禦，深夜亦不盲目縮容以消滅冷啟動風險。 |
| **🏆 Cost 敏感環境**<br>*(Non-Core Tier-3)* | `-42091.32` | **`-36765.50`**<br>`(成本優化)` | **AI 成功砍掉大量基礎設施浪費**。將全天 CPU 穩定壓榨在 60-80% 高效區間，並在海嘯期理性選擇短暫承受 700ms 延遲，以換取大幅降低雲端租金。 |

> 💡 **註：** 雖然總分帳面差距看似受限於固定背景基礎基礎租金（固定開銷基數大），但扣除不可避基本開銷後，AI 實質上消滅了傳統 HPA 因滯後震盪所引發的 **75% 營運浪費與所有致命大打結（HPA 尖峰延遲暴衝至 570ms）**。

---

## 📁 專案檔案結構

```text
├── Configurable_Scaling_Env.py   # 自訂 Gymnasium 微服務動態排隊模擬環境
├── DRL_Agent_Training.py         # SB3 PPO 演算法特
