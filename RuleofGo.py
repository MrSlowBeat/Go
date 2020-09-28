__author__ = 'JYH'


num_way = 8 #围棋路数
#深度优先搜索访问标记，若未访问为0，访问为1
visit = [[0 for i in range(num_way)] for j in range(num_way)]
isalive_flag = 0 #有“气”标志，有气为1，无气为0
player_flag = 1 # 当前玩家标志
suicide_flag = 0 # 自杀标志
checkerboard_data = [[ 0,  0,  0,  0,  0,  0,  0, 0],
                     [-1, -1, -1, -1, -1,  1, -1, 0],
                     [ 1,  1, -1,  1,  1,  1, -1, 0],
                     [-1,  1, -1,  1, -1,  1, -1, 0],
                     [ 1,  1,  1,  1, -1,  1,  1, 0],
                     [-1,  1, -1,  1, -1,  1, -1, 0],
                     [-1, -1, -1, -1, -1, -1, -1, 0],
                     [ 0,  0,  0,  0,  0,  0,  0, 0]]


def DFS(x, y):
    '''
    带记录的深度优先搜索
    用于判断棋子是否有气
    '''
    visit[x][y] = 1
    directions = [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]
    for dx, dy in directions:
        if dx < 0 or dx > num_way-1 or dy < 0 or dy > num_way-1:
            continue
        elif visit[dx][dy] == 0:
            if checkerboard_data[dx][dy] == 0:
                isalive_flag = 1
                return
            elif checkerboard_data[dx][dy] == - checkerboard_data[x][y]:
                continue
            elif checkerboard_data[dx][dy] == checkerboard_data[x][y]:
                DFS(dx, dy)
    # 以上条件都不满足,即所有路径都为死路,该棋子无“气”,停止搜索
    return


def clear_visit():
    '''清空搜索记录'''
    visit = [[0 for i in range(num_way)] for j in range(num_way)]
    isalive_flag = 0


def is_alive(x, y):
    '''有无“气”的判断，有气返回1，无气返回0'''
    # 清空搜索记录
    clear_visit()
    # 执行深度优先搜索
    DFS(x, y)
    return isalive_flag


def remove():
    '''提掉无气的棋子'''
    #"死子"列表
    token_list = []
    for i in range(num_way):
        for j in range(num_way):
            # 若当前位置是空,则直接跳过
            if checkerboard_data[i][j] == 0:
                continue
            # 判断对手的棋子有无“气”
            elif checkerboard_data[i][j] == - player_flag and is_alive(i, j) == 0:
                # 提掉
                token_list.append([i, j])
    # 若名单不为空,则提去名单中的所有棋子(仅对方棋子)
    if len(token_list) != 0:        
        for i, j in token_list:
            checkerboard_data[i][j] = 0
    # 自杀判定
    # 对方无“气”棋子全部提走后,对己方棋子进行有无“气”的判断,若己方仍存在无“气”棋子,则判定为自杀行为,自杀标志置1(因只需检测到一个无“气”子即说明是自杀,故无需继续检测,跳出循环)
    for i in range(num_way):
        for j in range(num_way):
            if checkerboard_data[i][j] == player_flag:
                if is_alive(i, j) == 0:
                	#自杀标志置1
                    suicide_flag = 1
                    break

