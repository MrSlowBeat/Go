'''
该程序为MuGo（一个极简AlphaGo的复现）的棋盘定义和规则实现的python源程序
后来参考minigo的代码:
1.在position类中增加了is_game_over方法，可以判断游戏是否结束
2.在position类中增加了all_legal_moves方法，可以返回所有合法落子点的mask

A board（棋盘） is a N x N numpy array.
A Coordinate（坐标） is a tuple(int, int) index into the board.
A Move（移动） is a (Coordinate c | None).
A PlayerMove（玩家的移动） is a (Color, Move) tuple

(0, 0) 为棋盘的左上角, and (18, 0) 为棋盘的左下角
'''


from collections import namedtuple
import copy
import itertools
import time
import numpy as np


# 用numpy array表示一个棋盘, 0是空, 1是黑, -1是白
# 这意味着交换颜色将数组乘以-1即可
WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)


class PlayerMove(namedtuple('PlayerMove', ['color', 'move'])):
    '''数据结构：玩家的移动：(颜色, 移动(int, int)) 元组'''
    pass


# 表示LibertyTracker对象中的“未找到组”
MISSING_GROUP_ID = -1


class IllegalMove(Exception):
    '''报错：非法的移动'''
    pass


# 这些变量由set_board_size函数初始化
N = None
ALL_COORDS = []
EMPTY_BOARD = None
NEIGHBORS = {}
DIAGONALS = {}


def set_board_size(n):
    '''
    初始化棋盘函数
    --------------------
    希望没有人同时运行9*9和19*19的游戏实例
    另外，永远也不要使用语句"from go import N, W, ALL_COORDS, EMPTY_BOARD"
    '''
    global N, ALL_COORDS, EMPTY_BOARD, NEIGHBORS, DIAGONALS
    if N == n:
        return
    N = n
    #所有的坐标
    ALL_COORDS = [(i, j) for i in range(n) for j in range(n)]
    #空白棋盘
    EMPTY_BOARD = np.zeros([n, n], dtype=np.int8)

    def check_bounds(c):
        '''检查是否超出边界，没有则返回1，否则返回0'''
        return c[0] % n == c[0] and c[1] % n == c[1]
    
    #[字典]每个坐标对应的值：邻居坐标，去除超出边界的点
    NEIGHBORS = {(x, y): list(filter(check_bounds, [(x+1, y), (x-1, y), (x, y+1), (x, y-1)])) for x, y in ALL_COORDS}
    #[字典]每个坐标对应的值：对角线坐标，去除超出边界的点
    DIAGONALS = {(x, y): list(filter(check_bounds, [(x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)])) for x, y in ALL_COORDS}


def place_stones(board, color, stones):
    '''放置棋子于棋盘上'''
    for s in stones:
        board[s] = color


def find_reached(board, c):
    '''
    使用带记录的深度优先搜索寻找所有连成一个整体的棋子（称为组），同时找到该组的边界
    输出：set(该棋子所属的整体),set(边界所有点)
    ----------------
    注意set()表示集合，有自动去重的功能
    '''
    color = board[c]
    chain = set([c])
    reached = set()
    frontier = [c]
    while frontier:
        current = frontier.pop()
        chain.add(current)
        for n in NEIGHBORS[current]:
            if board[n] == color and not n in chain:
                frontier.append(n)
            elif board[n] != color:
                reached.add(n)
    return chain, reached


def is_koish(board, c):
    '检查c（空点）是否被1个颜色包围，并返回该颜色'
    # 非空则返回None
    if board[c] != EMPTY:
        return None
    neighbors = {board[n] for n in NEIGHBORS[c]}
    # 周围只有1种颜色，且周围没有空点
    if len(neighbors) == 1 and not EMPTY in neighbors:
        return list(neighbors)[0]
    else:
        return None


def is_eyeish(board, c):
    '检查c是否为1个眼, 来限制MC rollouts'
    color = is_koish(board, c)
    # 若该空点周围不是由单颜色包围，则不是眼
    if color is None:
        return None
    #错误对角
    diagonal_faults = 0
    #该点的对角
    diagonals = DIAGONALS[c]
    if len(diagonals) < 4:
        diagonal_faults += 1
    for d in diagonals:
        #如果该对角不是邻居的同颜色或者空
        if not board[d] in (color, EMPTY):
            diagonal_faults += 1
    #错误对角数>1，则不是眼，返回空
    if diagonal_faults > 1:
        return None
    else:
        #否则是眼，返回颜色
        return color


class Group(namedtuple('Group', ['id', 'stones', 'liberties', 'color'])):
    '''
    stones:属于该组的坐标的set集合
    liberties:和该组邻接且为空点的坐标的set集合
    color:该组的颜色
    '''
    def __eq__(self, other):
        '''判断是否是同一个组'''
        return self.stones == other.stones and self.liberties == other.liberties and self.color == other.color


class LibertyTracker():
    '''自由点追踪器'''
    #静态方法：不使用该类的属性和方法
    @staticmethod
    def from_board(board):
        '''根据输入的棋盘board生成自由点跟踪器LibertyTracker'''
        #拷贝该棋盘
        board = np.copy(board)
        #当前组的id
        curr_group_id = 0
        # 再定义一个自由点跟踪器
        lib_tracker = LibertyTracker()
        for color in (WHITE, BLACK):
            #当棋盘上还有 黑白 棋子时
            while color in board:
                curr_group_id += 1
                #该颜色棋子的索引值
                found_color = np.where(board == color)
                #返回第一个该颜色棋子的坐标
                coord = found_color[0][0], found_color[1][0]
                #该棋子构成的组和其边界
                chain, reached = find_reached(board, coord)
                #找出边界中的自由点
                liberties = set(r for r in reached if board[r] == EMPTY)
                #生成组实例
                new_group = Group(curr_group_id, chain, liberties, color)
                #将其加入组字典
                lib_tracker.groups[curr_group_id] = new_group
                #将该组的id添加到该组所有点的[组索引]坐标上
                for s in chain:
                    lib_tracker.group_index[s] = curr_group_id
                #将已经遍历过的棋子标记为FILL
                place_stones(board, FILL, chain)

        #最大组id
        lib_tracker.max_group_id = curr_group_id

        #棋盘上每个点的自由点数
        liberty_counts = np.zeros([N, N], dtype=np.uint8)
        for group in lib_tracker.groups.values():
            #该组的自由点数
            num_libs = len(group.liberties)
            #将该自由点数赋给该组的每个点
            for s in group.stones:
                liberty_counts[s] = num_libs
        #自由点缓存
        lib_tracker.liberty_cache = liberty_counts
        #返回该自由点追踪器
        return lib_tracker

    def __init__(self, group_index=None, groups=None, liberty_cache=None, max_group_id=1):
        '''初始化函数'''
        # group_index:一个 N x N numpy array of 组id映射图，每个元素为组的id值，-1 表示没有组
        # groups:一个{组id:组实例}字典
        # liberty_cache: 一个 N x N numpy array of 自由点映射图（元素为棋盘上每个点的自由点数），0表示没有自由点
        self.group_index = group_index if group_index is not None else -np.ones([N, N], dtype=np.int16)
        self.groups = groups or {}
        self.liberty_cache = liberty_cache if liberty_cache is not None else np.zeros([N, N], dtype=np.uint8)
        self.max_group_id = max_group_id

    def __deepcopy__(self, memodict={}):
        '''深拷贝函数'''
        new_group_index = np.copy(self.group_index)
        new_lib_cache = np.copy(self.liberty_cache)
        new_groups = {
            group.id: Group(group.id, set(group.stones), set(group.liberties), group.color)
            for group in self.groups.values()
        }
        return LibertyTracker(new_group_index, new_groups, liberty_cache=new_lib_cache, max_group_id=self.max_group_id)

    def add_stone(self, color, c):
        '''在棋盘上添加一个子，返回提掉子的集合set{(int,int),...}'''
        #断言该点没有组
        assert self.group_index[c] == MISSING_GROUP_ID
        #提掉的子集合
        captured_stones = set()
        #敌人邻接组的id集合
        opponent_neighboring_group_ids = set()
        #盟友邻接组的id集合
        friendly_neighboring_group_ids = set()
        #空邻接组的坐标集合
        empty_neighbors = set()
        #对于c的每一个邻接点
        for n in NEIGHBORS[c]:
            #找到其组id
            neighbor_group_id = self.group_index[n]
            #非空组
            if neighbor_group_id != MISSING_GROUP_ID:
                #返回该组
                neighbor_group = self.groups[neighbor_group_id]
                if neighbor_group.color == color:
                    #盟友组
                    friendly_neighboring_group_ids.add(neighbor_group_id)
                else:
                    #敌人组
                    opponent_neighboring_group_ids.add(neighbor_group_id)
            else:
                #空组
                empty_neighbors.add(n)
        
        #将该点作为新组添加到groups、group_index、liberty_cache中，empty_neighbors中的坐标作为自由点
        new_group = self._create_group(color, c, empty_neighbors)

        #将每个所有盟友组与该新的组进行合并
        for group_id in friendly_neighboring_group_ids:
            new_group = self._merge_groups(group_id, new_group.id)

        for group_id in opponent_neighboring_group_ids:
            #对于每个敌人组
            neighbor_group = self.groups[group_id]
            #若改组的自由点数为1
            if len(neighbor_group.liberties) == 1:
                #提掉该组
                captured = self._capture_group(group_id)
                #更新被提子集合
                captured_stones.update(captured)
            else:
                #删除1个敌人组的自由点
                self._update_liberties(group_id, remove={c})
        
        #增加盟友组自由点
        self._handle_captures(captured_stones)

        #自杀式非法的
        if len(new_group.liberties) == 0:
            raise IllegalMove

        #返回提掉子的集合
        return captured_stones

    def _create_group(self, color, c, liberties):
        '''
        将新的组添加进groups、group_index、liberty_cache中
        返回新的组实例
        '''
        self.max_group_id += 1
        new_group = Group(self.max_group_id, set([c]), liberties, color)
        self.groups[new_group.id] = new_group
        self.group_index[c] = new_group.id
        self.liberty_cache[c] = len(liberties)
        return new_group

    def _merge_groups(self, group1_id, group2_id):
        '''将组1和组2进行合并,返回组1'''
        group1 = self.groups[group1_id]
        group2 = self.groups[group2_id]
        #去重地将g2元素添加进g1中
        group1.stones.update(group2.stones)
        #删除g2
        del self.groups[group2_id]
        #修改组id映射图
        for s in group2.stones:
            self.group_index[s] = group1_id
        
        #更新自由点映射图
        self._update_liberties(group1_id, add=group2.liberties, remove=(group2.stones | group1.stones))
        #返回g1
        return group1

    def _capture_group(self, group_id):
        '''提掉该组，返回所有被提掉的子的坐标set{(int,int)}'''
        dead_group = self.groups[group_id]
        #删除该组
        del self.groups[group_id]
        #将该组所有点置空
        for s in dead_group.stones:
            #组id图
            self.group_index[s] = MISSING_GROUP_ID
            #自由点图
            self.liberty_cache[s] = 0
        #返回所有被提掉的子
        return dead_group.stones

    def _update_liberties(self, group_id, add=None, remove=None):
        '''
        更新自由点映射图

        group_id为组1的id,
        add为组2中的自由点坐标,
        remove为g1和g2棋子的并集
        '''
        group = self.groups[group_id]
        if add:
            group.liberties.update(add)
        if remove:
            #difference_update移除两个集合都存在的元素
            group.liberties.difference_update(remove)
        #自由点数
        new_lib_count = len(group.liberties)
        #更新组1的自由点映射图
        for s in group.stones:
            self.liberty_cache[s] = new_lib_count

    def _handle_captures(self, captured_stones):
        '''处理所有被提子所造成的影响'''
        for s in captured_stones:
            for n in NEIGHBORS[s]:
                #对于每一个邻接组
                group_id = self.group_index[n]
                #若该组非空
                if group_id != MISSING_GROUP_ID:
                    #增加自由点
                    self._update_liberties(group_id, add={s})


class Position():
    def __init__(self, board=None, n=0, komi=7.5, caps=(0, 0), lib_tracker=None, ko=None, recent=tuple(), to_play=BLACK):
        '''
        初始化函数

        board: 一个 numpy array
        n: 一个表示到目前为止的moves个数（第几手）的 int
        komi: 一个 float, 代表第一个玩家给第二个玩家贴的目数
        caps: 一个 (int, int) 元组 of 提子数（该色棋子提掉对方棋子的个数） for B（黑子）, W（白子）
        lib_tracker: 一个 LibertyTracker 对象
        ko: 一个 Move
        recent: 一个由PlayerMoves构成的元组, 其中 recent[-1] 表示 the last move（上一个移动）.
        to_play: BLACK(int) 或 WHITE(int)，将要落的子的颜色
        '''
        #载入或新建棋盘
        self.board = board if board is not None else np.copy(EMPTY_BOARD)
        self.n = n
        self.komi = komi
        self.caps = caps
        #载入lib_tracker或根据棋盘构造lib_tracker
        self.lib_tracker = lib_tracker or LibertyTracker.from_board(self.board)
        self.ko = ko
        self.recent = recent
        self.to_play = to_play

    def __deepcopy__(self, memodict={}):
        '''深拷贝函数'''
        new_board = np.copy(self.board)
        new_lib_tracker = copy.deepcopy(self.lib_tracker)
        return Position(new_board, self.n, self.komi, self.caps, new_lib_tracker, self.ko, self.recent, self.to_play)

    def __str__(self, colors=True):
        '''对象打印函数，返回棋盘情况的字符串'''
        if colors:
            pretty_print_map = {
                WHITE: '\x1b[0;31;47mO',
                EMPTY: '\x1b[0;31;43m.',
                BLACK: '\x1b[0;31;40mX',
                FILL: '#',
                KO: '*',
            }
        else:
            pretty_print_map = {
                WHITE: 'O',
                EMPTY: '.',
                BLACK: 'X',
                FILL: '#',
                KO: '*',
            }
        board = np.copy(self.board)
        captures = self.caps
        #若ko非空，则放置ko棋子
        if self.ko is not None:
            place_stones(board, KO, [self.ko])
        #原始棋盘内容
        raw_board_contents = []
        for i in range(N):
            row = []
            for j in range(N):
                #指明上一步走的位置：如果上一个playermove存在，且坐标和其对应的move相同，则添加< ； 否则添加空格
                appended = '<' if (self.recent and (i, j) == self.recent[-1].move) else ' '
                row.append(pretty_print_map[board[i,j]] + appended)
                if colors:
                    row.append('\x1b[0m')
            raw_board_contents.append(''.join(row))
        #行标签 N~1
        row_labels = ['%2d ' % i for i in range(N, 0, -1)]
        #带标注的每一行的内容
        annotated_board_contents = [''.join(r) for r in zip(row_labels, raw_board_contents, row_labels)]
        #列标签
        header_footer_rows = ['   ' + ' '.join('ABCDEFGHJKLMNOPQRST'[:N]) + '   ']
        #带标注的棋盘
        #itertools.chain将不同容器中的元素连接起来，便于遍历
        annotated_board = '\n'.join(itertools.chain(header_footer_rows, annotated_board_contents, header_footer_rows))
        #细节：第n步，黑白提对方子的情况
        details = "\nMove: {}. Captures X: {} O: {}\n".format(self.n, *captures)
        return annotated_board + details

    def is_move_suicidal(self, move):
        '''
        判断该步是否为自杀

        如果没有自由点，则为自杀，返回True；否则返回False
        '''
        #潜在的自由点
        potential_libs = set()
        #对于每个邻接点
        for n in NEIGHBORS[move]:
            #该邻接点的组id
            neighbor_group_id = self.lib_tracker.group_index[n]
            #该点为空点
            if neighbor_group_id == MISSING_GROUP_ID:
                # 下完该步至少还有1个自由点，不算自杀
                return False
            #该邻接点的组
            neighbor_group = self.lib_tracker.groups[neighbor_group_id]
            #同色
            if neighbor_group.color == self.to_play:
                #将该组的自由点并入潜在的自由点
                potential_libs |= neighbor_group.liberties
            #如果是敌人组，且自由点数为1，则不是自杀
            elif len(neighbor_group.liberties) == 1:
                # 相反，这可以提掉敌人的组
                return False
        #将几个盟友组连接起来可能会造成自杀
        #当每个组只有1个自由点时

        #从潜在的自由点集合中移除该点
        potential_libs -= set([move])
        #如果没有自由点，则为自杀；否则不是
        return not potential_libs

    def is_move_legal(self, move):
        '''
        检查一个 move 是否合法
        检查该move是否在一个空点上, 且不是在ko上（局面重复）, 且不是自杀
        '''
        #若没有move操作，则合法
        if move is None:
            return True
        #该点不为空，则非法
        if self.board[move] != EMPTY:
            return False
        #该点是ko，则非法
        if move == self.ko:
            return False
        #该点是自杀，则非法
        if self.is_move_suicidal(move):
            return False

        return True
    
    def all_legal_moves(self):
        '''返回一个 np.ndarray, 尺寸为 N x N, 其中 1 = legal, 0 = illegal'''
        #默认每个棋子都是合法的
        legal_moves = np.ones([N, N], dtype=np.int8)
        #非空的不合法
        legal_moves[self.board != EMPTY] = 0
        #计算有4个邻接子的点
        #padding 是因为超出边缘的总是算作遗失的自由点
        adjacent = np.ones([N + 2, N + 2], dtype=np.int8)
        #有正常子的地方为1
        adjacent[1:-1, 1:-1] = np.abs(self.board)
        #棋盘每个点的邻接子数矩阵
        num_adjacent_stones = (adjacent[:-2, 1:-1] + adjacent[1:-1, :-2] +
                               adjacent[2:, 1:-1] + adjacent[1:-1, 2:])
        #Surrounded spots是空点且有4个邻接点的mask
        surrounded_spots = np.multiply(
            (self.board == EMPTY),
            (num_adjacent_stones == 4))
        #这些点可能是非法的，除非他们提到了敌人的子
        #反复迭代，手动检查每一个点
        #nonzero返回非零索引[[x1,x2,x3,...],[y1,y2,y3,...]]
        #transpose返回其转置[[x1,y1],[x2,y2],...]
        for coord in np.transpose(np.nonzero(surrounded_spots)):
            #如果是自杀，则标记为非法的
            if self.is_move_suicidal(tuple(coord)):
                legal_moves[tuple(coord)] = 0

        # ko点是非法的
        if self.ko is not None:
            legal_moves[self.ko] = 0

        # 弃子是合法的（该ndarray的最后一个？）
        # np.concatenate([legal_moves.ravel(), [1]])

        return legal_moves
    
    def pass_move(self, mutate=False):
        '''弃子，返回弃子后的position对象'''
        #若mutate（转换）开启则pos为position该对象，否则为对象的深拷贝
        pos = self if mutate else copy.deepcopy(self)
        pos.n += 1
        pos.recent += (PlayerMove(pos.to_play, None),)
        pos.to_play *= -1
        pos.ko = None
        return pos

    def flip_playerturn(self, mutate=False):
        '''翻转玩家：让敌人下棋，返回翻转后的position对象'''
        pos = self if mutate else copy.deepcopy(self)
        pos.ko = None
        pos.to_play *= -1
        return pos

    def get_liberties(self):
        '''获取自由点映射图'''
        return self.lib_tracker.liberty_cache

    def play_move(self, c, color=None, mutate=False):
        '''执行“下一步棋”的操作'''
        # 遵守CGOS游戏规则
        # 不能自杀
        # 中国/日韩 计分方式
        # Positional superko (目前是非常粗略的估计)

        #若颜色为空，则置为将要落下的子的颜色
        if color is None:
            color = self.to_play
        
        pos = self if mutate else copy.deepcopy(self)

        if c is None:
            #若move为空，则为弃子操作
            pos = pos.pass_move(mutate=mutate)
            return pos

        if not self.is_move_legal(c):
            #非法则报错
            raise IllegalMove()

        #放置一个棋子于棋盘上
        place_stones(pos.board, color, [c])
        #返回所提掉的对方的子集合
        captured_stones = pos.lib_tracker.add_stone(color, c)
        #将所提掉的位置置为空
        place_stones(pos.board, EMPTY, captured_stones)

        #敌人的颜色
        opp_color = color * -1

        #若提掉子只有1个，且落子点仅被敌人的子包围
        if len(captured_stones) == 1 and is_koish(self.board, c) == opp_color:
            #新的ko为被提子的坐标
            new_ko = list(captured_stones)[0]
        else:
            #否则没有ko现象
            new_ko = None

        if pos.to_play == BLACK:
            #如果是黑子下棋，则更新黑子提子数
            new_caps = (pos.caps[0] + len(captured_stones), pos.caps[1])
        else:
            #如果是白子下棋，则更新白子提子数
            new_caps = (pos.caps[0], pos.caps[1] + len(captured_stones))
        
        #更新当前的move数
        pos.n += 1
        #更新提子数
        pos.caps = new_caps
        #更新ko
        pos.ko = new_ko
        #记录该PlayerMove
        pos.recent += (PlayerMove(color, c),)
        #让对手下棋
        pos.to_play *= -1
        return pos
    
    def is_game_over(self):
        '''判断游戏是否结束，结束则返回1，否则返回0'''
        #move的个数大于2，且最后两步都是弃子
        return (len(self.recent) >= 2 and
                self.recent[-1].move is None and
                self.recent[-2].move is None)
    
    def score(self):
        '''计算终局得分情况'''
        #复制终局棋盘
        working_board = np.copy(self.board)
        #当终局棋盘存在空点时
        while EMPTY in working_board:
            #找到所有空点的坐标
            unassigned_spaces = np.where(working_board == EMPTY)
            c = unassigned_spaces[0][0], unassigned_spaces[1][0]
            #返回空点整体和边界
            territory, borders = find_reached(working_board, c)
            #边界的颜色集合
            border_colors = set(working_board[b] for b in borders)
            #黑子边界
            X_border = BLACK in border_colors
            #白子边界
            O_border = WHITE in border_colors
            #如果边界只有黑子没有白子，领域颜色为黑色
            if X_border and not O_border:
                territory_color = BLACK
            #如果边界只有白子没有黑子，领域颜色为白色
            elif O_border and not X_border:
                territory_color = WHITE
            else:
                #否则领域颜色为未知
                territory_color = UNKNOWN # dame, or seki
            
            #在领土上填满领域颜色棋子
            place_stones(working_board, territory_color, territory)
        
        #最终结果：黑子数-白子数-贴目数
        return np.count_nonzero(working_board == BLACK) - np.count_nonzero(working_board == WHITE) - self.komi

    def result(self):
        '''返回终局结果字符串'''
        score = self.score()
        if score > 0:
            #黑子赢了多少目
            return 'B+' + '%.1f' % score
        elif score < 0:
            #白子赢了多少目
            return 'W+' + '%.1f' % abs(score)
        else:
            #平局
            return 'DRAW'


class RandomRobot():
    '''机器人：随机走合法的落子点，而且不走自己的眼'''
    def select_move(self,pos,printer=True):
        '''选择动作'''
        #候选点列表
        candidates = []
        #所有可走的点索引
        legal_place = np.nonzero(pos.all_legal_moves())
        #转换成点元组
        for c in zip(legal_place[0],legal_place[1]):
            #不是自己的眼才走
            if is_eyeish(pos.board,c) != pos.to_play:
                candidates.append(c)

        #如果没有可以走的点，则返回None（弃子）
        if len(candidates)==0:
            if printer:
                print('player:' + ('BLACK' if pos.to_play == BLACK else 'WHITE'))
                print('pass move')
            return None
        else:
            #若有，则随机选择一个点进行移动
            random_index = np.random.choice(len(candidates))
            chosen_move = candidates[random_index]
            if printer:
                print('player:' + ('BLACK' if pos.to_play == BLACK else 'WHITE'))
                print('move:(%d,%d)' % (chosen_move[0],chosen_move[1]))
            return chosen_move


def play_one_game(pos, printer=True):
    '''玩一局游戏'''
    robot = RandomRobot()
    if printer:
        print("initial board")
        print(pos)
    while not pos.is_game_over():
        chosen_move = robot.select_move(pos,printer)
        pos.play_move(chosen_move, color=pos.to_play, mutate=True)
        if printer:
            print(pos)
        time.sleep(0.3)
    if printer:
        print("result:")
        print(pos.result()+'\n')


def main():
    set_board_size(9)
    p = Position()
    play_one_game(p)


if __name__ == '__main__':
    main()