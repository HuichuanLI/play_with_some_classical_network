import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


# 悬崖漫步环境
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.ncol = ncol  # 列数
        self.nrow = nrow  # 行数
        self.x = 0  # 当前x坐标
        self.y = nrow - 1  # 当前y坐标（初始在左下角起点）
        # 4种动作: 上(0)、下(1)、左(2)、右(3)
        self.change = [[0, -1], [0, 1], [-1, 0], [1, 0]]

    # 执行动作，返回下一状态、奖励、是否结束
    def step(self, action):
        # 执行动作，限制在网格内
        self.x = min(self.ncol - 1, max(0, self.x + self.change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + self.change[action][1]))
        # 一维状态编号
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        # 到达悬崖/终点区域
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            # 悬崖：奖励-100；终点：奖励-1
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    # 重置环境到起点
    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


# 打印智能体策略
def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            state = i * env.ncol + j
            if state in disaster:
                print('****', end=' ')
            elif state in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(state)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0.0 else 'o'
                print(pi_str, end=' ')
        print()


# Q-learning 算法
class QLearning:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([ncol * nrow, n_action])  # Q表初始化
        self.n_action = n_action
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率

    # epsilon-greedy 选择动作
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)  # 随机探索
        else:
            action = np.argmax(self.Q_table[state])  # 最优利用
        return action

    # 获取当前状态最优动作
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max:
                a[i] = 1
        return a

    # Q-learning 更新规则
    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0][a0]
        self.Q_table[s0][a0] += self.alpha * td_error


# ===================== 主程序：训练与测试 =====================
if __name__ == "__main__":
    # 环境设置：12列，4行 标准悬崖漫步
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)

    # Q-learning 超参数
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = QLearning(ncol, nrow, epsilon, alpha, gamma)

    # 训练参数
    num_episodes = 500  # 训练回合数
    return_list = []  # 记录每回合回报

    # 开始训练
    for i in tqdm(range(10)):  # 分10段显示进度
        for episode in range(num_episodes // 10):
            state = env.reset()
            done = False
            total_return = 0
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                # Q-learning 更新
                agent.update(state, action, reward, next_state)
                state = next_state
                total_return += reward
            return_list.append(total_return)

    # 绘制训练回报曲线
    plt.figure(figsize=(10, 6))
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Total Return')
    plt.title('Q-Learning on Cliff Walking')
    plt.show()

    # 打印最终策略
    action_meaning = ['^', 'v', '<', '>']
    # 悬崖位置（一维状态编号）
    cliff = list(range(37, 47))
    # 终点位置
    end = [47]
    print("最终Q-learning策略：")
    print_agent(agent, env, action_meaning, disaster=cliff, end=end)


