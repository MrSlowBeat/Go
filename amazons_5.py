import numpy as np
import torch
import time
from Storage import Log
from Game import AbstractArgs, AbstractGame


class Args(AbstractArgs):
    """
    参数类
    """
    def __init__(self):
        super().__init__()
        self.board_size = 5
        self.action_size = self.board_size ** 2
        self.num_max_layers = 3 * (self.board_size ** 2 - 8)
        self.GAME_NAME = 'amazons_5'
        # Mcts search parameters
        self.num_iter = 1000
        self.num_play_game = 20
        self.train_num_search = 1000
        self.random_num_search = 300
        self.sharpening_policy_t = 2
        self.smoothing_policy_t = 0.7
        self.smooth_policy_window = 3
        self.search_layers_threshold = 51
        self.Cpuct = 5
        # replay_buffer params
        self.N_threshold = 50
        self.N_Q_threshold = 1.5
        self.replay_decay_rate = 0.3
        self.replay_buffer_threshold = 5
        # NetWork params
        self.load_latest_model = False
        self.num_net = 3
        self.lr = 0.001
        self.lr_iter_threshold = 60
        self.weight_decay = 0.0001
        self.epochs = 10
        self.batch_size = 64
        self.num_params = 0
        # Process
        self.multiprocess = False
        self.num_process = 4
        self.SingleProcess = -1
        self.print_log_process = 0
        # Gpu Parallel
        self.cuda = torch.cuda.is_available()
        self.gpu_parallel = False
        self.gpu_num = torch.cuda.device_count()
        self.gpu_ids = range(0, torch.cuda.device_count(), 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.EPS = 1e-10
        # old and new networks PK
        self.Pk = False
        self.update_iter_threshold = 0.5
        self.num_pk = 10
        self.pk_step_threshold = 3
        self.start_pk_iter_threshold = 20


class Game(AbstractGame):
    """
    游戏规则类
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.BLACK = -2
        self.WHITE = 2
        self.EMPTY = 0
        self.ARROW = 1
        self.GAME_END = -1
        self.directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
        self.board_size = args.board_size
        self.step = [0, 1, 2]
        self.legal_pos = dict()
        self.action_size = args.board_size ** 2
        self.Cpuct = args.Cpuct
        self.N_DIR = 8
        self.board2d_shape = (self.board_size, self.board_size)
        self.board1d_shape = (self.action_size, )
        self.types = tuple(("queen", "next", "arrow"))
        self.flag = {"queen": 1, "next": 1, "arrow": -1}
        self.player = None
        self.board = None
        self.init_board()
        self.all_moves = None
        self.init_pre_table()

    def init_board(self):
        """
        棋盘初始化，定义初始棋盘和第一个玩家

        输出
        (numpy)棋盘，一维numpy数组
        """
        board_size = self.board_size
        self.player = self.WHITE
        self.board = np.zeros((board_size, board_size), dtype="int32")
        # BLACK
        self.board[0][board_size // 3] = self.BLACK
        self.board[0][2 * board_size // 3] = self.BLACK
        self.board[board_size // 3][0] = self.BLACK
        self.board[board_size // 3][board_size - 1] = self.BLACK
        # WHITE
        self.board[2 * board_size // 3][0] = self.WHITE
        self.board[2 * board_size // 3][board_size - 1] = self.WHITE
        self.board[board_size - 1][board_size // 3] = self.WHITE
        self.board[board_size - 1][2 * board_size // 3] = self.WHITE
        return self.board.reshape((-1,))


    def get_next_state(self, board, player, action):
        """
        一个真实步的三步，根据当前棋盘和动作转换到下一个棋盘

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
