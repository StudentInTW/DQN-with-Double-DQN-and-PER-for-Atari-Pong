# DQN with Double DQN and PER for Atari Pong

這是一個以 PyTorch 實作的深度強化學習專案。從最基本的 DQN 出發，先在 **CartPole-v1** 驗證演算法正確性，再擴展到以原始影像作為輸入的 **Atari Pong-v5**，最後整合 **Double DQN、Prioritized Experience Replay (PER)、Multi-step Return** 三種改良技術，並分析它們對樣本效率 (sample efficiency) 的影響。

完整實驗報告：[report.pdf](./report.pdf)

## 💡 核心技術與特色 (Features)
* **Vanilla DQN**：Experience Replay + Target Network，以 `gather()` 取出對應動作的 Q 值，Bellman target 在 `torch.no_grad()` 內計算，避免梯度流入 target network。
* **Atari 影像前處理**：灰階化 → 縮放至 84×84 → 堆疊最近 4 幀 (4×84×84)，讓 CNN Q-network 能從連續畫面推斷球的方向與速度。
* **Double DQN**：由 online network 選動作、target network 評估，降低 Q 值高估 (overestimation bias)。
* **Prioritized Experience Replay**：依 TD error 設定優先度 (α = 0.6, β = 0.4)，搭配 importance-sampling weights 修正抽樣偏差；新樣本以目前最大優先度插入。
* **Multi-step Return (n = 3)**：以 deque 暫存最近的轉移，讓稀疏的得分訊號更快傳遞，回合結束時把剩餘轉移全部寫回 buffer。
* **實驗追蹤**：使用 Weights & Biases 記錄 reward、episode length、epsilon 與評估結果；訓練支援從 checkpoint 接續 (`--resume-path`、`--start-env-count`)。

## 📂 專案架構 (Project Structure)
```text
├── dqn.py               # DQN / Enhanced DQN 訓練主程式（含所有可調參數）
├── fast_test.py         # 以多個 seed 快速評估模型平均分數
├── test_model.py        # 評估並錄製遊戲影片
├── checkpoints/         # 訓練好的模型權重
│   ├── LAB5_E84111201_task1.pt        # CartPole (Vanilla DQN)
│   ├── LAB5_E84111201_task2.pt        # Pong (Vanilla DQN)
│   └── LAB5_E84111201_task3_best.pt   # Pong (Enhanced DQN, best)
├── report.pdf           # 完整實驗報告（訓練曲線、分析與討論）
├── requirements.txt     # 套件依賴清單
└── README.md
```

## 🚀 快速開始 (Quick Start)

### 1. 安裝環境依賴
```bash
pip install -r requirements.txt
```

### 2. 開始訓練模型
```bash
# CartPole（預設環境）
python dqn.py --save-dir ./results_cartpole

# Atari Pong
python dqn.py --env-name ALE/Pong-v5 --save-dir ./results_pong
```

### 3. 評估模型（20 個 seed）
```bash
python fast_test.py --model-path checkpoints/LAB5_E84111201_task2.pt --episodes 20 --seed-start 0
```

## 📈 實驗結果 (Results)
所有結果皆為 20 個 seed 的評估平均：

| 任務 | 環境 | 平均分數 | 標準差 | 最低 / 最高 |
| --- | --- | --- | --- | --- |
| Vanilla DQN | CartPole-v1 | **481.35** | 37.82 | 371 / 500 |
| Vanilla DQN | Pong-v5 | **16.40** | 2.11 | 11 / 19 |
| Enhanced DQN (best) | Pong-v5 | **8.10** | 7.19 | −9 / 18 |

* Enhanced DQN 在 2.5M 環境步數內，從接近隨機的 −13 分提升到正分策略，2M 步之後進步最明顯。
* Vanilla DQN 的最終分數較高，主要是因為它經過多次 checkpoint 接續與低 epsilon 微調，兩者比較並非完全對照實驗；報告中有詳細分析與後續 ablation study 規劃。
