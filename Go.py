
from collections import namedtuple
import copy
import itertools
import numpy as np


class AbstractGame:
    pass


class PlayerMove(namedtuple('PlayerMove', ['color', 'move'])):
    '''数据结构：玩家的移动：(颜色, 移动(int, int)) 元组'''
    pass


class IllegalMove(Exception):
    '''报错：非法的移动'''
    pass


# 这些变量由set_board_size函数初始化
N = None
ALL_COORDS = []
EMPTY_BOARD = None
NEIGHBORS = {}
DIAGONALS = {}


class Game(AbstractGame):
    """
    游戏规则类
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 用numpy array表示一个棋盘, 0是空, 1是黑, -1是白
        # 这意味着交换颜色将数组乘以-1即可
        self.WHITE, self.EMPTY, self.BLACK, self.FILL, self.KO, self.UNKNOWN = range(-1, 5)
        # 表示LibertyTracker对象中的“未找到组”
        self.MISSING_GROUP_ID = -1

        self.board_size = args.board_size
        self.legal_pos = dict()
        self.action_size = args.board_size ** 2
        # self.Cpuct = args.Cpuct
        # self.N_DIR = 8
        # 棋盘的二维形状
        self.board2d_shape = (self.board_size, self.board_size)
        # 棋盘的一维形状
        self.board1d_shape = (self.action_size, )
        # self.types = tuple(("queen", "next", "arrow"))
        # self.flag = {"queen": 1, "next": 1, "arrow": -1}
        self.player = None
        self.board = None
        self.init_board()
        self.all_moves = None
        # self.init_pre_table()

    def init_board(self):
        """
        棋盘初始化，定义初始棋盘和第一个玩家(白棋)

        输出
        (numpy)棋盘，一维numpy数组
        """
        board_size = self.board_size
        self.player = self.BLACK
        self.board = np.zeros((board_size+2, board_size+2), dtype="int32")
        # 初始化超出边界的点
        for m in range(board_size+2):
            for n in range(board_size+2):
                if (m*n==0 or m==board_size+1 or n==board_size+1):
                    self.board[m][n]=self.OUT
        self.last_1_board = self.board.copy()
        self.last_2_board = self.board.copy()
        self.last_3_board = self.board.copy()
        return self.board.reshape((-1,))


    def get_next_state(self, board, player, action):
        """
        根据当前棋盘和所选择的动作转换到下一个棋盘

        输入
        board:(numpy) 目前的棋盘状态
        player:(int) 目前的玩家
        action:(int) 执行的动作
        输出
        board:(numpy) 下一个棋盘
        player:(int) 下一个玩家
        """
        assert type(board) is np.ndarray
        assert board.shape == self.board1d_shape
        assert player in (self.BLACK, self.WHITE)

        board = board.copy()

        assert board[action[0]] == player
        board[action[0]] = self.EMPTY
        assert board[action[1]] == self.EMPTY
        board[action[1]] = player
        assert board[action[2]] == self.EMPTY
        board[action[2]] = self.ARROW

        return board, -player

    def get_legal_action(self, board, layers, start_pos=None):
        """
        根据当前的棋盘和游戏规则来获取合法的动作

        输入
        board:(numpy) 当前棋盘
        layers:(int) MCTs的深度
        start_pos:(int) 执行动作的起始位置
        输出
        legal actions:(list) 所有可移动到的位置
        """
        types = self.types[layers % 3]
        if types == "queen":
            key = self.to_string(board)
            if key not in self.legal_pos:
                assert type(board) is np.ndarray
                assert board.shape == self.board1d_shape
                all_queens = np.where(board == self.WHITE)[0]
                queens = []
                # 选Queen
                for q in all_queens:
                    # 选方向
                    for moves in self.all_moves[q]:
                        if board[moves[0]] == self.EMPTY:
                            queens.append(q)
                            break
                self.legal_pos[key] = queens
            else:
                queens = self.legal_pos[key]
            return queens

        elif types == "next":
            key = self.to_string(board) + self.to_string(np.array(start_pos, dtype=np.int))
            if key not in self.legal_pos:
                assert type(board) is np.ndarray
                assert board.shape == self.board1d_shape
                if type(start_pos) not in (np.int, np.int32, np.int64):
                    print(f"type(start_pos) = {type(start_pos)}")
                assert type(start_pos) in (np.int, np.int32, np.int64)
                # print(board.reshape(5, 5), start_pos)
                assert board[start_pos] == self.WHITE
                nexts = []
                for moves in self.all_moves[start_pos]:
                    for move in moves:
                        if board[move] == self.EMPTY:
                            nexts.append(move)
                        else:
                            break
                self.legal_pos[key] = nexts
            else:
                nexts = self.legal_pos[key]
            return nexts

        else:
            key = self.to_string(board) + self.to_string(np.array(start_pos, dtype=np.int))
            if key not in self.legal_pos:
                assert type(board) is np.ndarray
                assert board.shape == self.board1d_shape
                assert type(start_pos) in (np.int32, np.int, np.int64), print(f'type(next_pos)={type(start_pos)},next_pos={start_pos}')
                assert board[start_pos] == self.WHITE

                arrows = []
                for moves in self.all_moves[start_pos]:
                    for move in moves:
                        if board[move] == self.EMPTY:
                            arrows.append(move)
                        else:
                            break
                self.legal_pos[key] = arrows
            else:
                arrows = self.legal_pos[key]
            return arrows

    def game_over(self, board, player):
        """
        判断游戏是否结束

        输入
        board:(numpy) 当前棋盘
        player:(int) 当前玩家
        输出
        game_state:(int) 0/-1  游戏没有结束/游戏结束
        """
        assert type(board) is np.ndarray
        assert board.shape == self.board1d_shape
        my_queens = np.where(board == player)[0]
        for queen in list(my_queens):
            assert board[queen] == player
            for moves in self.all_moves[queen]:
                if board[moves[0]] == self.EMPTY:
                    return 0
        return self.GAME_END
