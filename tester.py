import copy
 
class Move():
  def  __init__(self, point = None, is_pass = False, is_resign = False):
    assert(point is not None) ^is_pass ^is_resign
    self.point = point
    #是否轮到我下
    self.is_play (self.pint is not None)
    self.is_pass = is_pass
    self.is_resign = is_resign
    
  @classmethod
  def  play(cls, point):
    return Move(point = point)
  
  @classmethod
  #让对方继续下
  def  pass_turn(cls):
    return move(is_pass = True)
  
  @classmethod
  #投子认输
  def  resign(cls):
    return move(is_resign = True)