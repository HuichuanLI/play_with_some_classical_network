import numpy as np


class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.action_space = 4
        self.reset()
        self.P = self.build_P()  # 状态转移矩阵（必须有！）

    def build_P(self):
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # 上、下、左、右

        for y in range(self.nrow):
            for x in range(self.ncol):
                s = y * self.ncol + x
                # 终点/悬崖：不动
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

                    # 悬崖惩罚
                    if new_y == self.nrow - 1 and new_x > 0 and new_x != self.ncol - 1:
                        reward = -100
                        done = True
                    # 终点
                    if new_y == self.nrow - 1 and new_x == self.ncol - 1:
                        done = True

                    P[s][a] = [(1.0, next_s, reward, done)]
        return P

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


# --------------------- 策略迭代算法（Policy Iteration）---------------------
class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta  # 收敛阈值
        self.gamma = gamma
        self.state_num = env.ncol * env.nrow
        self.action_num = 4

        self.v = [0.0] * self.state_num  # 价值函数
        # 初始化策略：每个动作均匀随机
        self.pi = [[1 / self.action_num for _ in range(self.action_num)] for _ in range(self.state_num)]

    # 策略评估：固定策略，计算对应价值函数
    def policy_evaluation(self):
        cnt = 0
        while True:
            max_diff = 0
            new_v = [0.0] * self.state_num

            for s in range(self.state_num):
                v = 0
                # 按当前策略执行所有动作
                for a, prob_a in enumerate(self.pi[s]):
                    for p, next_s, r, done in self.env.P[s][a]:
                        v += prob_a * p * (r + self.gamma * self.v[next_s] * (1 - done))
                new_v[s] = v
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            self.v = new_v
            cnt += 1
            if max_diff < self.theta:
                break
        return cnt

    # 策略提升：根据价值函数更新为贪婪策略
    def policy_improvement(self):
        policy_stable = True  # 策略是否稳定

        for s in range(self.state_num):
            old_action = np.argmax(self.pi[s])  # 旧策略最优动作

            # 计算所有Q(s,a)
            qsa_list = []
            for a in range(self.action_num):
                qsa = 0
                for p, next_s, r, done in self.env.P[s][a]:
                    qsa += p * (r + self.gamma * self.v[next_s] * (1 - done))
                qsa_list.append(qsa)

            # 新贪婪策略
            max_q = max(qsa_list)
            best_cnt = qsa_list.count(max_q)
            self.pi[s] = [1 / best_cnt if q == max_q else 0.0 for q in qsa_list]

            # 策略是否变化
            if np.argmax(self.pi[s]) != old_action:
                policy_stable = False
        return policy_stable

    # 主迭代流程
    def policy_iteration(self):
        cnt = 0
        while True:
            eval_cnt = self.policy_evaluation()
            stable = self.policy_improvement()
            cnt += 1
            print(f"第 {cnt} 次策略迭代，策略评估迭代 {eval_cnt} 轮")
            if stable:
                break
        print(f"策略迭代完成！共 {cnt} 轮策略迭代")


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

    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()

    print("\n策略迭代最终策略：")
    print_agent(agent, action_meaning, disaster=list(range(37, 47)), end=[47])
