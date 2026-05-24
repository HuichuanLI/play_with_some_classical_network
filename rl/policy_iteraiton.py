import copy
from typing import List, Tuple


class CliffWalkingEnv:
    """
    悬崖漫步（Cliff Walking）强化学习环境
    标准网格环境：4行×12列，底部行为悬崖，右下角为目标点
    动作空间：0=上，1=下，2=左，3=右
    奖励规则：普通步-1，掉落悬崖-100，到达目标0
    """
    # 动作常量定义，提升代码可读性
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    def __init__(self, ncol: int = 12, nrow: int = 4):
        """
        初始化环境
        :param ncol: 网格列数，默认12
        :param nrow: 网格行数，默认4
        """
        self.ncol = ncol
        self.nrow = nrow
        self.state_num = nrow * ncol  # 总状态数，便于外部调用
        self.action_num = 4  # 固定动作数量

        # 状态转移矩阵 P[状态][动作] = [(概率, 下一状态, 奖励, 结束标志)]
        self.P = self._create_transition_matrix()

    def _create_transition_matrix(self) -> List[List[List[Tuple]]]:
        """创建环境状态转移矩阵（私有方法，外部不调用）"""
        # 初始化转移矩阵：[总状态数][动作数]
        transition = [[[] for _ in range(self.action_num)] for _ in range(self.state_num)]

        # 动作坐标偏移量 [上, 下, 左, 右]
        action_delta = [
            [0, -1],  # 上：y坐标减1
            [0, 1],  # 下：y坐标加1
            [-1, 0],  # 左：x坐标减1
            [1, 0]  # 右：x坐标加1
        ]

        # 遍历所有网格位置
        for y in range(self.nrow):
            for x in range(self.ncol):
                current_state = y * self.ncol + x

                # 遍历所有动作
                for action in range(self.action_num):
                    # 当前位置是悬崖/目标点，任何动作都无奖励且终止
                    if y == self.nrow - 1 and x > 0:
                        transition[current_state][action] = [(1.0, current_state, 0, True)]
                        continue

                    # 计算下一步位置（边界限制，防止越界）
                    delta_x, delta_y = action_delta[action]
                    next_x = max(0, min(self.ncol - 1, x + delta_x))
                    next_y = max(0, min(self.nrow - 1, y + delta_y))
                    next_state = next_y * self.ncol + next_x

                    # 初始化奖励和终止状态
                    reward = -1
                    done = False

                    # 判断下一步是否到达悬崖或终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        # 掉落悬崖（非终点）
                        if next_x != self.ncol - 1:
                            reward = -100

                    transition[current_state][action] = [(1.0, next_state, reward, done)]

        return transition


class PolicyIteration:
    """
    策略迭代（Policy Iteration）算法实现
    包含：策略评估 + 策略提升 + 迭代主循环
    """

    def __init__(self, env, theta: float, gamma: float):
        """
        初始化策略迭代算法
        :param env: 强化学习环境（必须包含状态转移矩阵 P）
        :param theta: 策略评估收敛阈值
        :param gamma: 折扣因子
        """
        self.env = env
        self.state_num = env.nrow * env.ncol  # 总状态数
        self.action_num = 4  # 动作数量固定为4

        # 初始化状态价值函数 V(s)
        self.v = [0.0] * self.state_num

        # 初始化策略 π(s,a)：均匀随机策略
        self.pi = [[1 / self.action_num for _ in range(self.action_num)]
                   for _ in range(self.state_num)]

        # 算法超参数
        self.theta = theta
        self.gamma = gamma

    def policy_evaluation(self) -> None:
        """策略评估：固定策略，更新状态价值函数 V(s) 直至收敛"""
        iteration_cnt = 0  # 迭代计数器

        while True:
            max_diff = 0.0  # 记录最大价值变化量
            new_v = [0.0] * self.state_num  # 新的价值函数

            # 遍历所有状态
            for state in range(self.state_num):
                q_value_sum = 0.0  # 存储当前状态的价值 V(s)

                # 遍历所有动作，计算 Q(s,a)
                for action in range(self.action_num):
                    q_sa = 0.0
                    # 遍历环境状态转移
                    for prob, next_state, reward, done in self.env.P[state][action]:
                        # 贝尔曼方程计算动作价值
                        q_sa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))

                    # 累积当前策略下的状态价值
                    q_value_sum += self.pi[state][action] * q_sa

                new_v[state] = q_value_sum
                # 更新最大差值
                max_diff = max(max_diff, abs(new_v[state] - self.v[state]))

            # 更新价值函数
            self.v = new_v
            iteration_cnt += 1

            # 收敛判断
            if max_diff < self.theta:
                break

        print(f"策略评估进行 {iteration_cnt} 轮后完成")

    def policy_improvement(self) -> List[List[float]]:
        """策略提升：根据当前价值函数，贪心更新策略 π(s,a)"""
        for state in range(self.state_num):
            # 计算当前状态下所有动作的 Q(s,a)
            q_values = []
            for action in range(self.action_num):
                q_sa = 0.0
                for prob, next_state, reward, done in self.env.P[state][action]:
                    q_sa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                q_values.append(q_sa)

            # 贪心策略：取最大 Q 值对应的动作
            max_q = max(q_values)
            max_action_count = q_values.count(max_q)  # 最优动作数量

            # 最优动作均分概率，非最优动作概率为0
            self.pi[state] = [1 / max_action_count if q == max_q else 0.0 for q in q_values]

        print("策略提升完成")
        return self.pi

    def policy_iteration(self) -> None:
        """策略迭代主循环：交替执行评估与提升，直至策略收敛"""
        while True:
            # 1. 策略评估
            self.policy_evaluation()

            # 2. 深拷贝旧策略，用于比较
            old_policy = copy.deepcopy(self.pi)

            # 3. 策略提升
            new_policy = self.policy_improvement()

            # 4. 策略不再变化，迭代结束
            if old_policy == new_policy:
                print("策略已收敛，迭代完成！")
                break


def print_agent(agent, action_meaning, disaster=None, end=None):
    """
    打印强化学习智能体的状态价值函数与策略可视化结果
    :param agent: 训练完成的策略迭代智能体
    :param action_meaning: 动作含义列表，如 ['^', 'v', '<', '>']
    :param disaster: 悬崖/障碍状态列表，默认空列表
    :param end: 目标终点状态列表，默认空列表
    """
    # 设置默认值，避免可变默认参数陷阱
    if disaster is None:
        disaster = []
    if end is None:
        end = []

    env = agent.env
    nrow, ncol = env.nrow, env.ncol

    # 打印状态价值
    print("=" * 80)
    print("📊 状态价值 V(s)")
    print("=" * 80)
    for i in range(nrow):
        for j in range(ncol):
            state = i * ncol + j
            value = agent.v[state]
            print(f"{value:6.3f}", end=" ")  # 统一格式化输出
        print()

    print("\n" + "=" * 80)
    print("🎯 最优策略 π(s)")
    print("=" * 80)

    # 打印策略
    for i in range(nrow):
        for j in range(ncol):
            state = i * ncol + j
            if state in disaster:
                print("****", end="  ")  # 悬崖/危险区域
            elif state in end:
                print("EEEE", end="  ")  # 终点
            else:
                pi = agent.pi[state]
                # 生成策略字符串：有概率的动作显示符号，无概率显示 o
                pi_str = "".join([action_meaning[k] if pi[k] > 0 else "o"
                                  for k in range(len(action_meaning))])
                print(pi_str, end="  ")
        print()
    print("=" * 80)


# ====================== 运行测试（与之前代码无缝衔接）======================
if __name__ == "__main__":
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']  # 上、下、左、右
    theta = 0.001
    gamma = 0.9

    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()

    # 悬崖状态 37~46，终点 47
    print_agent(agent, action_meaning, disaster=list(range(37, 47)), end=[47])
