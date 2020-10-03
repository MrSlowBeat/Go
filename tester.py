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

def to_python(player_state):
    if player_state is None:
        return 'None'
    if player_state == Player.black:
        return Player.black
    return Player.white


# 棋盘大小
board_size = 19
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