import numpy as np


# --------------------- 悬崖漫步环境（已支持动态规划 P 表）---------------------
class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.action_space = 4  # 上下左右
        self.reset()

        # 动态规划必备：状态转移概率表 P[state][action] = [(prob, next_state, reward, done)]
        self.P = self.build_P()

    def build_P(self):
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]
        # 动作：0上 1下 2左 3右
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        for y in range(self.nrow):
            for x in range(self.ncol):
                s = y * self.ncol + x
                # 终点/悬崖 无法再移动
                if y == self.nrow - 1 and x > 0:
                    for a in range(4):
                        P[s][a] = [(1.0, s, 0, True)]
                    continue

                for a in range(4):
                    dx, dy = change[a]
                    new_x = min(self.ncol - 1, max(0, x + dx))
                    new_y = min(self.nrow - 1, max(0, y + dy))
                    next_s = new_y * self.ncol + new_x
                    done = False
                    reward = -1

                    # 掉落悬崖
                    if new_y == self.nrow - 1 and new_x > 0 and new_x != self.ncol - 1:
                        reward = -100
                        done = True
                    # 到达终点
                    if new_y == self.nrow - 1 and new_x == self.ncol - 1:
                        done = True

                    P[s][a] = [(1.0, next_s, reward, done)]
        return P

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


# --------------------- 价值迭代算法（优化版）---------------------
class ValueIteration:
    """价值迭代算法（优化版，无冗余、无语法错）"""

    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta  # 收敛阈值
        self.gamma = gamma  # 折扣因子
        self.state_num = env.ncol * env.nrow
        self.action_num = 4

        # 初始化价值函数 & 策略
        self.v = [0.0] * self.state_num
        self.pi = [[] for _ in range(self.state_num)]

    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0.0
            new_v = [0.0] * self.state_num

            # 遍历所有状态
            for s in range(self.state_num):
                qsa_list = []
                # 计算当前状态 s 下所有动作的 Q(s,a)
                for a in range(self.action_num):
                    qsa = 0.0
                    for prob, next_s, r, done in self.env.P[s][a]:
                        qsa += prob * (r + self.gamma * self.v[next_s] * (1 - done))
                    qsa_list.append(qsa)

                # 价值迭代核心：直接取最大 Q 作为新 V
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            # 更新价值
            self.v = new_v
            cnt += 1

            # 收敛判断
            if max_diff < self.theta:
                break

        print(f"价值迭代完成，共迭代 {cnt} 轮")
        # 导出最终策略
        self.get_policy()

    def get_policy(self):
        """根据最优价值函数导出贪婪策略"""
        for s in range(self.state_num):
            qsa_list = []
            for a in range(self.action_num):
                qsa = 0.0
                for prob, next_s, r, done in self.env.P[s][a]:
                    qsa += prob * (r + self.gamma * self.v[next_s] * (1 - done))
                qsa_list.append(qsa)

            max_q = max(qsa_list)
            best_count = qsa_list.count(max_q)
            # 多个最优动作均分概率
            self.pi[s] = [1 / best_count if q == max_q else 0.0 for q in qsa_list]


# --------------------- 打印策略函数 ---------------------
def print_agent(agent, action_meaning, disaster=[], end=[]):
    env = agent.env
    for i in range(env.nrow):
        for j in range(env.ncol):
            s = i * env.ncol + j
            if s in disaster:
                print('****', end=' ')
            elif s in end:
                print('EEEE', end=' ')
            else:
                pi = agent.pi[s]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if pi[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


# --------------------- 主程序运行 ---------------------
if __name__ == "__main__":
    env = CliffWalkingEnv(ncol=12, nrow=4)
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9

    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()

    print("\n价值迭代最终策略：")
    print_agent(agent, action_meaning, disaster=list(range(37, 47)), end=[47])
