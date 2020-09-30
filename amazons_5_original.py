import numpy as np
import torch
import time
from Storage import Log
from Game import AbstractArgs, AbstractGame


class Args(AbstractArgs):
    """
    Parameters class
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
    Game rules' class
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
        Checkerboard initialization, which defines the initial chessboard and the first player

        :return board:(numpy) One dimensional numpy array
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

    def play_one_game(self, proc, num_iter, mcts):
        """
        Play a complete game of chess

        :param proc:(int) process number, -1 represents a single process
        :param num_iter:(int) the number of iterations
        :param mcts:(object) Monte Carlo tree search object
        :returns player: (int) the loser
                 return_data: (list) training data sampled by MCTs
                 trajectory: (list) the real trajectory of a game
        """
        return_data, trajectory = [], []
        board = self.init_board()
        player = self.player
        z = 0
        play_step = 0
        while z == 0:
            ts = time.time()
            if proc == self.args.SingleProcess or proc == self.args.print_log_process:
                Log.print_step_num(play_step)
                Log.print_board(board.reshape(self.board_size, self.board_size))
            # Unified use of white chess pieces to search and simulate
            transformed_board = self.change_perspectives(board, player)
            if (3 * play_step) >= self.args.search_layers_threshold:
                next_action, data, trajectory_data = mcts.select_action(proc, transformed_board, num_iter, 3 * play_step, "sharpening policy")
            elif (3 * play_step) < (self.args.search_layers_threshold - self.args.smooth_policy_window):
                next_action, data, trajectory_data = mcts.select_action(proc, transformed_board, num_iter, 3 * play_step, "normal policy")
            else:
                next_action, data, trajectory_data = mcts.select_action(proc, transformed_board, num_iter, 3 * play_step, "smoothing policy")
                # print("smoothing policy")
            return_data.extend(data)
            trajectory.extend(trajectory_data)
            te = time.time()
            if proc == self.args.SingleProcess or proc == self.args.print_log_process:
                Log.print_action_and_search_time(proc, player, next_action, ts, te)
            board, player = self.get_next_state(board, player, next_action)
            z = self.game_over(board, player)
            play_step += 1
            if z == self.GAME_END:
                if proc == self.args.SingleProcess or proc == self.args.print_log_process:
                    Log.print_game_result(player)
                    Log.print_board(board.reshape(self.board_size, self.board_size))
                return player, return_data, trajectory

    def get_next_state(self, board, player, action):
        """
        Three steps for a real move, convert to the next chessboard according to the current chessboard and action

        :param board:(numpy) the current chessboard
        :param player:(int) the current player
        :param action:(int) the next move
        :returns board:(numpy)  the converted chessboard
                 player:(int) next player
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
        According to the current board and game rules to obtain legal actions

        :param board:(numpy) the current board
        :param layers:(int) depth of the mcts tree
        :param start_pos:(int) Starting point of action
        return legal actions:(list) All movable points
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
        Judge whether the game is over

        :param board:(numpy) the current board
        :param player:(int) the current player
        :return game_state:(int) 0/-1  The game is not over/The game is over
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

    def change_perspectives(self, board, player):
        """
        Change the perspective when playing black chess (because the neural network always uses the perspective of white chess training)

        :param board:(numpy) the current board
        :param player:(int) the current player
        :return board:(numpy) converted board
        """
        if player == self.WHITE:
            return board

        b = np.copy(board)
        for i in range(self.action_size):
            if b[i] == self.WHITE or b[i] == self.BLACK:
                b[i] = -b[i]
        return b

    def to_string(self, board):
        """
        Convert chessboard to string

        :param board:(numpy) the current board
        :return string:(str) string corresponding to chessboard
        """
        return board.tostring()

    def board_flip(self, board, pi, pos=None):
        """
        Make the chessboard diagonally and flip it symmetrically to increase the amount of data

        :param board:(numpy) the current board
        :param pi:(list) posterior policy
        :param pos:(int) selected move
        :return board_list:(list) list data after flipped in four directions
        """
        board.shape = self.board2d_shape
        board_list = []
        new_pos = None
        # 1
        new_b = np.reshape(board, self.board2d_shape)
        pi_ = np.reshape(pi, self.board2d_shape)
        if pos is None:
            board_list += [(new_b, list(pi_.ravel()))]
        else:
            new_pos = np.zeros_like(pi_)
            new_pos[pos // self.board_size][pos % self.board_size] = 1
            board_list += [(new_b, new_pos, list(pi_.ravel()))]
        # 2
        new_b = np.fliplr(new_b)
        new_pi = np.fliplr(pi_)
        if pos is None:
            board_list += [(new_b, list(new_pi.ravel()))]
        else:
            new_pos = np.fliplr(new_pos)
            board_list += [(new_b, new_pos, list(new_pi.ravel()))]
        # 3
        new_b = np.flipud(new_b)
        new_pi = np.flipud(new_pi)
        if pos is None:
            board_list += [(new_b, list(new_pi.ravel()))]
        else:
            new_pos = np.flipud(new_pos)
            board_list += [(new_b, new_pos, list(new_pi.ravel()))]
        # 4
        new_b = np.fliplr(new_b)
        new_pi = np.fliplr(new_pi)
        if pos is None:
            board_list += [(new_b, list(new_pi.ravel()))]
        else:
            new_pos = np.fliplr(new_pos)
            board_list += [(new_b, new_pos, list(new_pi.ravel()))]

        board.shape = self.board1d_shape
        return board_list

    def init_pre_table(self):
        """
        Initialize the information table according to the rules of the game
        """
        # Converting 1-D coordinates to 2-D coordinates
        xy_of = np.array([(i, j) for i in range(self.board_size) for j in range(self.board_size)], dtype=np.int)

        # 8 directions
        delta_1d = np.array((-self.board_size - 1, -self.board_size, -self.board_size + 1,
                             -1, +1,
                             +self.board_size - 1, +self.board_size, +self.board_size + 1),
                            dtype=np.int)

        delta_2d = np.array(((-1, -1), (-1, 0), (-1, +1),
                             (0, -1), (0, +1),
                             (+1, -1), (+1, 0), (+1, +1)),
                            dtype=np.int)

        # Define the predication table (by direction, the empty table is removed)
        self.all_moves = [[] for _ in range(self.action_size)]

        for s in range(self.action_size):
            for d in range(self.N_DIR):
                move_list = []
                for i in range(self.board_size):
                    cur_x, cur_y = xy_of[s] + delta_2d[d] * (i + 1)
                    cur = s + delta_1d[d] * (i + 1)
                    if cur_x not in range(self.board_size) or cur_y not in range(self.board_size):
                        break
                    move_list.append(cur)

                if move_list:
                    self.all_moves[s].append(tuple(move_list))
        self.all_moves = tuple(self.all_moves)

    def get_next_board(self, board, next_a, layers, start_a=None):
        """
        Each small step transition (will produce a non real state board)

        :param board:(numpy) the current board
        :param next_a:(int) putting point
        :param layers:(int) the depth of mcts tree
        :param start_a:(int) start point
        :return b:(numpy) converted chessboard
        """
        assert type(board) is np.ndarray
        assert board.shape == self.board1d_shape
        if self.types[layers % 3] == "queen":
            board = board.copy()
            return board
        elif self.types[layers % 3] == "next":
            b = board.copy()
            assert b[start_a] == self.WHITE
            b[start_a] = self.EMPTY
            assert b[next_a] == self.EMPTY
            b[next_a] = self.WHITE
            return b
        else:
            b = board.copy()
            assert b[next_a] == self.EMPTY
            b[next_a] = self.ARROW
            b = self.change_perspectives(b, self.BLACK)
            return b
