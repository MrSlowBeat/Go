import enum
import time
import random
from collections import namedtuple
import copy
 
 
class Player(enum.Enum):
    '''玩家类（黑或白）'''
    black = 1
    white = 2
 
    @property
    def other(self):
        '''返回对方棋子颜色，如果本方是白棋，那就返回Player.black'''
        if self == Player.white:
            return Player.black
        else:
            return Player.white
 
 
class Point(namedtuple('Point', 'row col')):
  def  neighbors(self):
    '''
    返回当前点的相邻点，也就是相对于当前点的上下左右四个点
    '''
    return [
        Point(self.row - 1, self.col),
        Point(self.row + 1, self.col),
        Point(self.row, self.col - 1),
        Point(self.row, self.col + 1),
    ]
 
 
class Move():
    def __init__(self, point=None, is_pass=False, is_resign=False):
        # ^表示异或，意味着三种情况只有其中一种成立
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        # 是否轮到我下
        self.is_play = (self.point is not None)
        # 是否弃子
        self.is_pass = is_pass
        # 是否投降
        self.is_resign = is_resign
 
    @classmethod
    def play(cls, point):
        '''下一步棋'''
        return Move(point=point)
 
    @classmethod
    def pass_turn(cls):
        '''让对方继续下'''
        return Move(is_pass=True)
 
    @classmethod
    def resign(cls):
        '''投子认输'''
        return Move(is_resign=True)
 
 
class GoString():
    '''一片棋子类'''
    def __init__(self, color, stones, liberties):
        self.color = color   #黑/白
        # 两个集合为immutable类型，不可以改变
        self.stones = frozenset(stones)  #stone就是棋子
        self.liberties = frozenset(liberties)  #自由点
    
    def without_liberty(self, point):
        '''将某点从自由点中去除掉'''
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)
 
    def with_liberty(self, point):
        '''将某点添加到自由点'''
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)
 
    def merged_with(self, go_string):
        '''
        落子之后，两片相邻棋子可能会合成一片

        假设*代表黑棋，o代表白棋，x代表没有落子的棋盘点，当前棋盘如下：
        x  x  x  x  x  x
        x  *  x! *  o  *
        x  x  x  *  o  x
        x  x  *  x  o  x
        x  x  *  o  x  x
        注意看带!的x，如果我们把黑子下在那个地方，那么x!左边的黑棋和新下的黑棋会调用当前函数进行合并，
        同时x!上方的x和下面的x就会成为合并后相邻棋子共同具有的自由点。同时x!原来属于左边黑棋的自由点，
        现在被一个黑棋占据了，所以下面代码要把该点从原来的自由点集合中去掉
        '''
        # 确定颜色和己方相同
        assert go_string.color == self.color
        # 将棋子连接起来
        combined_stones = self.stones | go_string.stones
        # 返回连接成一片的棋子
        return GoString(self.color, combined_stones,
                        (self.liberties | go_string.liberties) - combined_stones)
    
    @property
    def num_liberties(self):
        '''该片棋子自由点的数量'''
        return len(self.liberties)
 
    def __eq__(self, other):
        '''该片棋子和other（另一片棋子）是否相等'''
        return isinstance(other,
                          GoString) and self.color == other.color and self.stones == other.stones and self.liberties == other.liberties
 
 
class Board():
    '''实现棋盘'''
    def __init__(self, num_rows, num_cols):
        # 行数和列数
        self.num_rows = num_rows
        self.num_cols = num_cols
        # 棋盘表格
        self._grid = {}
        # 为空棋盘添加hash值
        self._hash = zobrist_EMPTY_BOARD
 
    def zobrist_hash(self):
        '''返回该盘棋局的hash值'''
        return self._hash
 
    def place_stone(self, player, point):
        '''放置一枚棋子'''
        # 确保位置在棋盘内
        assert self.is_on_grid(point)
        # 确定给定位置没有被占据
        assert self._grid.get(point) is None
 
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []
 
        for neighbor in point.neighbors():
            # 判断落子点上下左右的邻接点情况
            if not self.is_on_grid(neighbor):
                continue
 
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                # 如果邻接点没有被占据，那么就是当前落子点的自由点
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    # 记录与棋子同色的连接棋子
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    # 记录落点邻接点与棋子不同色的棋子
                    adjacent_opposite_color.append(neighbor_string)
 
        # 将当前落子与棋盘上相邻的棋子合并成一片
        # 先用new_string构造一片棋子对象
        new_string = GoString(player, [point], liberties)

        # 相同颜色合并
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)
        for new_string_point in new_string.stones:
            # 访问棋盘某个点时返回与该点棋子相邻的所有棋子集合
            self._grid[new_string_point] = new_string

        # 增加落子的hash值记录
        self._hash ^= zobrist_HASH_CODE[point, None] # 和空子异或（第一次是置空，后面遇到后是撤销空）
        self._hash ^= zobrist_HASH_CODE[point, player] # 和当前子颜色异或

        for other_color_string in adjacent_opposite_color:
            # 当该点被占据前，它属于反色棋子的自由点，占据后就不再属于反色棋子自由点
            # 去掉该子占据的自由点，将对手棋子添加到棋盘中
            replacement = other_color_string.without_liberty(point)
            # 如果对手棋子还有自由点
            if replacement.num_liberties:
                self._replace_string(other_color_string.without_liberty(point))
            else:
                # 如果落子后，相邻反色棋子的所有自由点都被堵住，对方棋子被吃掉
                self._remove_string(other_color_string)
    
    def _replace_string(self, new_string):
        '''将对手棋子添加到棋盘中'''
        for point in new_string.stones:
            self._grid[point] = new_string
 
    def is_on_grid(self, point):
        '''是否在棋盘内'''
        return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols
 
    def get(self, point):
        '''返回该片棋子的颜色'''
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color
 
    def get_go_string(self, point):
        '''返回该片棋子'''
        string = self._grid.get(point)
        if string is None:
            return None
        return string
 
    def _remove_string(self, string):
        '''从棋盘上删除一片棋子'''
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    # 如果该棋子（将要被删除）的邻居棋子是对手棋子，则将该点重新添加为对手棋子的自由点
                    self._replace_string(neighbor_string.with_liberty(point))
            # 将该点清空
            self._grid[point] = None
            # 由于棋子被拿掉后，对应位置状态发生变化，因此修改编码
            self._hash ^= zobrist_HASH_CODE[point, string.color] #和该色棋子异或（撤销该子）
            self._hash ^= zobrist_HASH_CODE[point, None] #和空子异或（置为空子）


class GameState():
    '''棋盘状态的检测和落子检测'''
    def __init__(self, board, next_player, previous, move):
        # 当前棋盘（Board）
        self.board = board
        # 当前玩家（Player）
        self.next_player = next_player
        # 上一个状态（GameState）
        self.previous_state = previous
        # 上一个移动（Move）
        self.last_move = move
        if previous is None:
            # 以前的状态为空，创建以前的状态为空
            self.previous_states = frozenset()
        else:
            # 以前的状态 = 上一个状态前的所有状态 + 上一个状态
            self.previous_states = frozenset(previous.previous_states | {(previous.next_player,
                                                                          previous.board.zobrist_hash())})

    def apply_move(self, move):
        '''执行一个移动'''
        if move.is_play:
            # 轮到我下，下一个棋盘为下一个棋
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            # 弃子或者投降
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)
 
    @classmethod
    def new_game(cls, board_size):
        '''创建一个新棋局'''
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
 
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)
 
    def is_over(self):
        '''游戏是否结束'''
        if self.last_move is None:
            # 游戏的第一步，即上一步为空
            return False
        if self.last_move.is_resign:
            # 一方认输
            return True
        # 上一步
        second_last_move = self.previous_state.last_move
        # 上一步的上一步为空，即上一步为游戏的第一步
        if second_last_move is None:
            return False
        # 如果两个棋手同时放弃落子，棋局结束
        return self.last_move.is_pass and second_last_move.is_pass
 
    def is_move_self_capture(self, player, move):
        '''判断是否为自己吃自己'''
        #如果不是轮到自己下，则返回否
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        # 先落子，完成吃子后再判断是否是自己吃自己
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0
 
    @property
    def situation(self):
        '''情况：返回下一个玩家和当前棋盘'''
        return (self.next_player, self.board)
 
    def does_move_violate_ko(self, player, move):
        '''判断是否是KO'''
        #如果不是轮到自己下，则返回否
        if not move.is_play:
            return False
 
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        #下一个情况
        next_situation = (player.other, next_board)
 
        # 判断Ko不仅仅看是否返回上一步的棋盘而是检测是否返回以前有过的棋盘状态
        # 修改,我们不用在循环检测，只要看当前数值与前面数值是否匹配即可
        return (next_situation in self.previous_states)
 
    def is_valid_move(self, move):
        '''判断该移动是否合法'''
        #游戏结束，则不合法
        if self.is_over():
            return False
        #若投降或棋子，则合法
        if move.is_pass or move.is_resign:
            return True
        #该点没有棋子 且 不是自己吃自己 且 不是KO
        return (self.board.get(move.point) is None and
                not self.is_move_self_capture(self.next_player, move) and
                not self.does_move_violate_ko(self.next_player, move))
 
 
def is_point_an_eye(board, point, color):
    '''判断是否为眼'''
    # 若该点有棋子，则不是眼
    if board.get(point) is not None:
        return False
    
    for neighbor in point.neighbors():
        # 检测邻接点全是己方棋子
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False
    # 四个对角线位置至少有三个被己方棋子占据
    friendly_corners = 0
    # 超出边界的点
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1)
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    if off_board_corners > 0:
        # 除了超出边界的点都是己方的点
        return off_board_corners + friendly_corners == 4
    # 没有超出边界的点，己方的点要超过3个
    return friendly_corners >= 3


class Agent:
    def __init__(self):
        pass
    def select_move(self, game_state):
        raise NotImplementedError()


class RandomBot(Agent):
    def select_move(self, game_state):
        '''
        遍历棋盘，只要看到一个不违反规则的位置就落子
        '''
        candidates = []
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)
                # 是合法的棋子而且不是眼
                if game_state.is_valid_move(Move.play(candidate)) and not \
                        is_point_an_eye(game_state.board,
                                        candidate,
                                        game_state.next_player):
                    candidates.append(candidate)
        # 如果没有可下的点，则弃子
        if not candidates:
            return Move.pass_turn()
        
        # 在所有可选位置随便选一个
        return Move.play(random.choice(candidates))


def print_move(player, move):
    '''打印如何移动'''
    if move.is_pass:
        #弃子
        move_str = 'passes'
    elif move.is_resign:
        #投降
        move_str = 'resign'
    else:
        #正常移动：打印列行值
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))


def print_board(board):
    '''打印棋盘'''
    for row in range(board.num_rows, 0, -1):
        bump = ' ' if row <= 9 else ''
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
 
    print('   ' + ' '.join(COLS[:board.num_cols]))


def to_python(player_state):
    if player_state is None:
        return 'None'
    if player_state == Player.black:
        return Player.black
    return Player.white

 
# 棋盘的列用字母表示
COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    Player.black: 'x',
    Player.white: 'o'
}


# 棋盘大小
board_size = 9
# 用一个64位整型对应每个棋盘
MAX63 = 0x7fffffffffffffff
# 3*19*19 / MAX63
# 发明这种编码算法的人叫zobrist
zobrist_HASH_CODE = {}
# 空棋盘的hash值
zobrist_EMPTY_BOARD = 0
 
for row in range(1,board_size+1):
    for col in range(1,board_size+1):
        for state in (None,Player.black,Player.white):
            # 随机选取一个整数对应当前位置,这里默认当前取随机值时不会与前面取值发生碰撞
            code = random.randint(0, MAX63)
            zobrist_HASH_CODE[Point(row, col), state] = code

print('HASH_CODE = {')
for (pt, state), hash_code in zobrist_HASH_CODE.items():
  print(' (%r, %s): %r,' % (pt, to_python(state), hash_code))

print('}')
print(' ')
print('EMPTY_BOARD = %d' % (zobrist_EMPTY_BOARD,))


def main():
    # 初始化9*9棋盘
    global board_size
    game = GameState.new_game(board_size)
    bots = {
        Player.black: RandomBot(),
        Player.white: RandomBot()
    }

    while not game.is_over():
        time.sleep(0.3)
        # 清屏幕
        print(chr(27) + "[2J")
        # 打印棋盘
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)


if __name__ == '__main__':
    main()