---
title: "第0章：全局地图"
nav_order: 1
description: "一页纸看懂强化学习：Bellman 方程是核心，所有算法都是近似求解它的不同工具"
---

> **读法**：初读时作为导览，了解各章位置和联系；读完全书后再回来，会发现每一行都有了新的分量。

---

## 强化学习要解决什么问题

一个**智能体（Agent）**在环境中反复行动，每步得到一个奖励信号。目标是找到一个**策略 $\pi$**，使得长期折扣累积奖励的期望最大：

$$\pi^* = \arg\max_\pi \; \mathbb{E}_\pi \!\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

这个问题的完整数学描述是**马尔可夫决策过程（MDP）**：$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$，其中 $\mathcal{P}(s'|s,a)$ 是状态转移概率，$\mathcal{R}(s,a,s')$ 是奖励函数。

---

## 核心：Bellman 方程

RL 的理论中心是两组 Bellman 方程。**几乎所有算法都是在以不同方式近似求解它们。**

**Bellman 期望方程**（评估给定策略 $\pi$）：

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a)\bigl[\mathcal{R}(s,a,s') + \gamma V^\pi(s')\bigr]$$

**Bellman 最优方程**（直接寻找最优策略）：

$$V^*(s) = \max_a \sum_{s'} \mathcal{P}(s'|s,a)\bigl[\mathcal{R}(s,a,s') + \gamma V^*(s')\bigr]$$

两个方程的差别只有一处：期望方程对动作**加权平均**（按策略 $\pi$），最优方程对动作**取最大值**。

```text
当前状态价值 = 即时奖励 + γ × 下一状态价值（的期望 / 最大值）
                                         ↑                ↑
                                    期望方程           最优方程
```

---

## 各章的位置：从理论到实践

```text
第1章   RL 是什么？        直觉与全局观，RL vs 监督学习
第2章   数学工具箱          概率、期望、梯度——本书用到的数学
第3章   MDP               RL 的数学语言：S A P R γ，V Q 函数定义
第4章   动态规划           已知模型时精确求解 Bellman 方程：策略迭代、值迭代
        ↓  
        模型未知时，以下方法从采样中近似求解 Bellman 方程
        ↓
第5章   蒙特卡洛 & TD      MC：回合结束后用真实回报更新
                           TD：每步用"当前估计"自举更新（Bootstrapping）
第6章   Q-Learning & Sarsa  最简单的无模型求解：直接逼近 Q*
第7章   DQN                神经网络近似 Q 函数，突破维度灾难
        ↓
        从"学 Q 函数"转向"直接优化策略"
        ↓
第8章   策略梯度           Policy Gradient 定理，REINFORCE
第9章   Actor-Critic       Actor（策略）+ Critic（Bellman 期望方程）联合训练
第10章  PPO               当前机器人 RL 主力：Clip 限制步长，稳定训练
第11章  算法全景图         Model-Based / Off-Policy / 最大熵 SAC / 离线 RL
        ↓
        走出仿真
        ↓
第12章  Sim-to-Real       域随机化、系统辨识、Actuator Net——把仿真策略部署到真机
第13章  进阶路径           论文清单与后续学习地图
```

---

## 算法分类一览

| 类别 | 核心思路 | 代表算法 | 本书位置 |
| --- | --- | --- | --- |
| **动态规划** | 已知 $\mathcal{P}$，直接迭代求解 Bellman 方程 | 策略迭代、值迭代 | 第4章 |
| **蒙特卡洛** | 用完整轨迹的真实回报 $G_t$ 估计价值 | MC Control | 第5章 |
| **时序差分** | 用单步 Bootstrapping 估计，无需等回合结束 | TD(0)、Sarsa | 第5章 |
| **值函数（无模型）** | 直接学 $Q^*$，用 $\varepsilon$-贪心执行策略 | Q-Learning、DQN | 第6–7章 |
| **策略梯度** | 直接对策略参数 $\theta$ 做梯度上升 | REINFORCE、PPO、TRPO | 第8、10章 |
| **Actor-Critic** | Critic 近似 Bellman，为 Actor 提供基线 | A2C、SAC、TD3、DDPG | 第9、11章 |
| **Model-Based** | 先学环境模型，再在模型中规划 | Dyna、MBPO、DreamerV3 | 第11章 |

---

## 一句话串联全书

$$\underbrace{\text{MDP}}_{\text{第3章}} \xrightarrow{\text{已知模型}} \underbrace{\text{动态规划}}_{\text{第4章}} \xrightarrow{\text{去掉模型}} \underbrace{\text{TD / Q-Learning}}_{\text{第5–6章}} \xrightarrow{\text{加神经网络}} \underbrace{\text{DQN}}_{\text{第7章}} \xrightarrow{\text{换优化目标}} \underbrace{\text{策略梯度 / PPO}}_{\text{第8–10章}} \xrightarrow{\text{上真机}} \underbrace{\text{Sim-to-Real}}_{\text{第12章}}$$

每一步演进都有一个明确的动机：**当前方法的局限**，正好是下一章要解决的问题。
