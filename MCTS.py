

"""
AlphaGOZero中的蒙特卡洛树搜索的python代码实现
"""


import numpy as np
import copy 


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """
    定义MCTS中一个树的结点类
    每个结点存储其父节点sp,儿子结点sc,结点访问数N(s),动作价值Q(s,a),先验概率P(s,a),和基于访问数N(s)计算的先验自适应分数U
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # {"a":TreeNode}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """
        通过创建新的儿子节点扩展树
        action_priors：策略函数的输出，list(tuple(a,p(s,a)))
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        选择儿子节点中使 PUCT = Q(s,a)+U 最大的
        返回：
        tuple(a,sc)
        """
        #max(TreeNode(sc).get_value(c_puct))
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        根据叶子节点的评估值v(sl)更新该结点Q(s,a),同时更新访问次数N(s)
        """
        # 增加访问次数
        self._n_visits += 1
        # 增量平均更新，Q更新完为所有访问过的叶子节点的v(sl)的平均值
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        递归更新所有祖先结点
        """
        # 如果不是根节点，应该先更新其父节点
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回该节点的价值PUCT
        一个“叶子节点的评估值v(sl), Q(s,a), 和该节点基于N(s,a)的先验自适应分数U”的结合
        c_puct：一个在(0, inf)区间的数，控制相关参数Q(s,a),P(s,a)等在该结点得分上的影响力
        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """
        检查是否为叶子节点
        该结点下没有节点被扩展进来
        """
        return self._children == {}

    def is_root(self):
        """
        检查是否为根节点
        """
        return self._parent is None


class MCTS(object):
    """
    一个蒙特卡洛树搜索的简单实现
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        参数
        policy_value_fn：一个函数需要输入(s)，输出list(tuple(a, p(s,a))),v(s)∈[-1, 1]
            (根据当前玩家的视角得到的游戏结束时的期望得分)
        c_puct：一个在(0, inf)区间上的数，控制探索收敛到最大价值max(v(sl))策略的速度, 值越高越依赖于先验概率（探索）
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        # 模拟次数
        self._n_playout = n_playout

    def _playout(self, state):
        """
        从根节点到叶子节点运行一次模拟, 得到叶子节点的价值v(sl)且将结果反向传播给其祖先结点
        状态空间的修改是内嵌的，所以必须提供一份拷贝。
        参数：
        state：状态s的一个拷贝
        """
        node = self._root
        while True:
            if node.is_leaf():
                break             
            # 贪婪地选择下一个移动
            action, node = node.select(self._c_puct)
            # 更新状态s<-sc      
            state.do_move(action)

        # 使用神经网络评估叶子节点，输出 list(tuple(a, p(s,a))) 和一个当前玩家的得分 v in [-1, 1]
        action_probs, leaf_value = self._policy(state)

        # 检查模拟是否结束
        end, winner = state.game_end()
        # 若未结束，则继续扩展
        if not end:
            node.expand(action_probs)
        else:
            # 对于终止状态，返回"真"叶子结点的值
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0
                 
        #更新此遍历中节点的值Q(s,a)和访问次数N(s)
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        按顺序运行模拟返回可行的a和相应的p(s,a)

        参数：
        state：当前状态, 包括游戏状态和当前玩家
        temp：温度参数，在(0, 1]区间上控制探索等级
        返回：
        可行的 a 和相应的p(s,a)
        """        
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
  
        # 基于根节点对其访问的次数N(s,a)来计算概率分布
        #list(tuple(a,N(s,a) ) )
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        #归一化的a的概率分布
        act_probs = softmax(1.0/temp * np.log(visits))       
        #返回可行的tuple(a)和相应的概率分布tuple(p(s,a) )
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        在树中向下走，保留所有已经探索过的子树
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        """打印MCTS"""
        return "MCTS"


class MCTSPlayer(object):
    """基于MCTS的AI玩家"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        #是否为自玩
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        """设置玩家索引"""
        self.player = p

    def reset_player(self):
        """重置玩家"""
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        """获取动作"""
        # 所有合法的移动
        sensible_moves = board.availables
        # 初始化每个移动的概率P(s,a)
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            #通过MCTS获得每个移动的概率
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            #如果是自玩
            if self._is_selfplay:
                # 在探索过程中增加 Dirichlet 噪声(自玩训练的需要）
                # 根据下面的概率公式选择动作
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新MCTS的根节点，重新利用搜索树
                self.mcts.update_with_move(move)
            else:
                # 使用默认的温度参数temp=1e-3，几乎等价于选择概率P(s,a)最大的动作
                move = np.random.choice(acts, p=probs)
                # 重置MCTS根节点
                self.mcts.update_with_move(-1)
                # location = board.move_to_location(move)
                # print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                #返回移动，移动概率P(s,a)
                return move, move_probs
            else:
                #返回移动
                return move
        else:
            print("警告：棋盘已满")

