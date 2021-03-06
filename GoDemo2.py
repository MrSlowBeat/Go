#!/usr/bin/python3


'''
该程序实现了围棋的大部分规则，但没有实现终局和判定胜负的规则
该程序的优点在于围棋游戏的可视化做的非常好
该程序的缺点在于数据结构优化较差，用于训练模型则效率可能比较低
'''


# 使用Python内置GUI模块tkinter
from tkinter import *
# ttk覆盖tkinter部分对象，ttk对tkinter进行了优化
from tkinter.ttk import *
# 深拷贝时需要用到copy模块
import copy
import tkinter.messagebox


class Application(Tk):
	'''围棋应用对象定义'''
	def __init__(self,my_mode_num=9):
		'''初始化棋盘,默认九路棋盘'''
		Tk.__init__(self)
		# 模式，九路棋：9，十三路棋：13，十九路棋：19
		self.mode_num=my_mode_num
		# 窗口尺寸设置，默认：1.8
		self.size=1.8
		# 棋盘每格的边长
		self.dd=360*self.size/(self.mode_num-1)
		# 相对九路棋盘的矫正比例
		self.p=1 if self.mode_num==9 else (2/3 if self.mode_num==13 else 4/9)
		# 定义棋盘阵列,超过边界：-1，无子：0，黑棋：1，白棋：2，[行，列]
		self.positions=[[0 for i in range(self.mode_num+2)] for i in range(self.mode_num+2)]
		# 初始化棋盘，所有超过边界的值置-1
		for m in range(self.mode_num+2):
			for n in range(self.mode_num+2):
				if (m*n==0 or m==self.mode_num+1 or n==self.mode_num+1):
					self.positions[m][n]=-1
		# 拷贝三份棋盘“快照”，悔棋和判断“打劫”时需要作参考
		self.last_3_positions=copy.deepcopy(self.positions)
		self.last_2_positions=copy.deepcopy(self.positions)
		self.last_1_positions=copy.deepcopy(self.positions)
		# 记录鼠标经过的地方，用于显示shadow时
		self.cross_last=None
		# 当前轮到的玩家，黑：0，白：1，执黑先行
		self.present=0 
		# 初始停止运行，点击“开始游戏”运行游戏
		self.stop=True
		# 悔棋次数，次数大于0才可悔棋，初始置0（初始不能悔棋），悔棋后置0，下棋或弃手时恢复为1，以禁止连续悔棋
		self.regretchance=0
		# 图片资源，存放在当前目录下的/Pictures/中
		self.photoW=PhotoImage(file = "./Pictures/W.png")
		self.photoB=PhotoImage(file = "./Pictures/B.png")
		self.photoBD=PhotoImage(file = "./Pictures/"+"BD"+"-"+str(self.mode_num)+".png")
		self.photoWD=PhotoImage(file = "./Pictures/"+"WD"+"-"+str(self.mode_num)+".png")
		self.photoBU=PhotoImage(file = "./Pictures/"+"BU"+"-"+str(self.mode_num)+".png")
		self.photoWU=PhotoImage(file = "./Pictures/"+"WU"+"-"+str(self.mode_num)+".png")
		# 用于黑白棋子图片切换的列表，U代表待落下的棋子，D代表已经落下的棋子
		self.photoWBU_list=[self.photoBU,self.photoWU]
		self.photoWBD_list=[self.photoBD,self.photoWD]
		# 窗口大小
		self.geometry(str(int(600*self.size))+'x'+str(int(400*self.size)))
		# 画布控件，作为容器，bg代表背景颜色
		self.canvas_bottom=Canvas(self,bg='#369',bd=0,width=600*self.size,height=400*self.size)
		# place为放置的位置
		self.canvas_bottom.place(x=0,y=0)
		# 几个功能按钮
		self.startButton=Button(self,text='开始游戏',command=self.start)
		self.startButton.place(x=480*self.size,y=200*self.size)
		self.passmeButton=Button(self,text='弃一手',command=self.passme)
		self.passmeButton.place(x=480*self.size,y=225*self.size)	
		self.regretButton=Button(self,text='悔棋',command=self.regret)
		self.regretButton.place(x=480*self.size,y=250*self.size)
		# 初始悔棋按钮禁用
		self.regretButton['state']=DISABLED
		self.replayButton=Button(self,text='重新开始',command=self.reload)
		self.replayButton.place(x=480*self.size,y=275*self.size)
		# 更改游戏模式按钮，如果当前是9路，则显示13路，否则显示9路
		self.newGameButton1=Button(self,text=('十三' if self.mode_num==9 else '九')+'路棋',command=self.newGame1)
		self.newGameButton1.place(x=480*self.size,y=300*self.size)
		# 更改游戏模式按钮，如果当前是19路，则显示13路，否则显示19路
		self.newGameButton2=Button(self,text=('十三' if self.mode_num==19 else '十九')+'路棋',command=self.newGame2)
		self.newGameButton2.place(x=480*self.size,y=325*self.size)
		self.quitButton=Button(self,text='退出游戏',command=self.quit)
		self.quitButton.place(x=480*self.size,y=350*self.size)
		# 画棋盘，填充颜色，参数（左上角x，左上角y，右下角，右下角y，填充颜色），其中x对应棋盘的列，y对应棋盘的行
		self.canvas_bottom.create_rectangle(0*self.size,0*self.size,400*self.size,400*self.size,fill='#c51')
		# 刻画棋盘线及九个点
		# 先画外框粗线
		self.canvas_bottom.create_rectangle(20*self.size,20*self.size,380*self.size,380*self.size,width=3)
		# 棋盘上的九个定位点，以中点为模型，移动位置，以作出其余八个点
		for m in [-1,0,1]:
			for n in [-1,0,1]:
				self.oringinal=self.canvas_bottom.create_oval(200*self.size-self.size*2,200*self.size-self.size*2,
				200*self.size+self.size*2,200*self.size+self.size*2,fill='#000')
				self.canvas_bottom.move(self.oringinal,m*self.dd*(2 if self.mode_num==9 else (3 if self.mode_num==13 else 6)),
				n*self.dd*(2 if self.mode_num==9 else (3 if self.mode_num==13 else 6)))
		# 画中间的线条，横着7条，竖着7条
		for i in range(1,self.mode_num-1):
			self.canvas_bottom.create_line(20*self.size,20*self.size+i*self.dd,380*self.size,20*self.size+i*self.dd,width=2)
			self.canvas_bottom.create_line(20*self.size+i*self.dd,20*self.size,20*self.size+i*self.dd,380*self.size,width=2)
		# 放置右侧初始图片(太极图)
		self.pW=self.canvas_bottom.create_image(500*self.size+11, 65*self.size,image=self.photoW)
		self.pB=self.canvas_bottom.create_image(500*self.size-11, 65*self.size,image=self.photoB)
		# 每张图片都添加image标签，方便reload函数删除图片
		self.canvas_bottom.addtag_withtag('image',self.pW)
		self.canvas_bottom.addtag_withtag('image',self.pB)
		# 鼠标移动时，调用shadow函数，显示随鼠标移动的棋子
		self.canvas_bottom.bind('<Motion>',self.shadow)
		# 鼠标左键单击时，调用getdown函数，放下棋子
		self.canvas_bottom.bind('<Button-1>',self.getDown)
		# 设置退出快捷键<Ctrl>+<D>，快速退出游戏
		self.bind('<Control-KeyPress-d>',self.keyboardQuit)
	
	def start(self):
		'''开始游戏函数，点击“开始游戏”时调用'''
		# 删除右侧太极图
		self.canvas_bottom.delete(self.pW)
		self.canvas_bottom.delete(self.pB)
		# 利用右侧图案提示开始时谁先落子
		if self.present==0:
			self.create_pB()
			self.del_pW()
		else:
			self.create_pW()
			self.del_pB()
		# 开始标志，解除stop
		self.stop=None
	
	def passme(self):
		'''放弃一手函数，跳过落子环节'''
		# 悔棋恢复
		if not self.regretchance==1:
			self.regretchance+=1
		else:
			self.regretButton['state']=NORMAL
		# 拷贝棋盘状态，记录前三次棋局
		# 拷贝3 <- 拷贝2，拷贝3保存上上上一次棋局
		self.last_3_positions=copy.deepcopy(self.last_2_positions)
		# 拷贝2 <- 拷贝1，拷贝2保存上上一次棋局
		self.last_2_positions=copy.deepcopy(self.last_1_positions)
		# 拷贝1 <- 当前棋盘，拷贝1保存上一次棋局
		self.last_1_positions=copy.deepcopy(self.positions)
		# 删除image_added_sign
		self.canvas_bottom.delete('image_added_sign')
		# 轮到下一玩家
		if self.present==0:
			self.create_pW()
			self.del_pB()
			self.present=1
		else:
			self.create_pB()
			self.del_pW()
			self.present=0
	
	def regret(self):
		'''
		悔棋函数，可悔棋一回合，下两回合不可悔棋
		使用拷贝3恢复上上上一回合棋局，当前棋局变为上上上一棋局
		恢复完毕后，将上上上一棋局保存到拷贝1中，清除拷贝2,3
		'''
		# 判定是否可以悔棋，以前第三盘棋局复原棋盘
		if self.regretchance==1:
			# 本局悔棋后下一局不能悔棋，置数、禁用按钮
			self.regretchance=0
			self.regretButton['state']=DISABLED
			list_of_b=[]
			list_of_w=[]
			self.canvas_bottom.delete('image')
			if self.present==0:
				self.create_pB()
			else:
				self.create_pW()
			# 清空当前棋盘
			for m in range(1,self.mode_num+1):
				for n in range(1,self.mode_num+1):
					self.positions[m][n]=0
			# 使用棋盘拷贝3进行记录
			for m in range(len(self.last_3_positions)):
				for n in range(len(self.last_3_positions[m])):
					# 记录黑白棋子的[列,行]值
					if self.last_3_positions[m][n]==1:
						list_of_b+=[[n,m]]
					elif self.last_3_positions[m][n]==2:
						list_of_w+=[[n,m]]
			# 恢复所有白棋子和黑棋子的位置和对应图片
			self.recover(list_of_b,0)
			self.recover(list_of_w,1)
			# 拷贝1 <- 拷贝3
			self.last_1_positions=copy.deepcopy(self.last_3_positions)
			# 将拷贝2,3清空
			for m in range(1,self.mode_num+1):
				for n in range(1,self.mode_num+1):
					self.last_2_positions[m][n]=0
					self.last_3_positions[m][n]=0
	
	def reload(self):
		'''重新加载函数,删除图片，序列归零，设置一些初始参数，点击“重新开始”时调用'''
		# 停止游戏
		if self.stop==1:
			self.stop=0
		# 删除所有图片
		self.canvas_bottom.delete('image')
		# 清除后悔标志
		self.regretchance=0
		# 当前玩家重置为黑子
		self.present=0
		self.create_pB()
		# 重置棋盘和其相应的拷贝
		for m in range(1,self.mode_num+1):
			for n in range(1,self.mode_num+1):
				self.positions[m][n]=0
				self.last_3_positions[m][n]=0
				self.last_2_positions[m][n]=0
				self.last_1_positions[m][n]=0
	
	def create_pW(self):
		'''太极图白色区域的创建'''
		self.pW=self.canvas_bottom.create_image(500*self.size+11, 65*self.size,image=self.photoW)
		self.canvas_bottom.addtag_withtag('image',self.pW)
	
	def create_pB(self):
		'''太极图黑色区域的创建'''
		self.pB=self.canvas_bottom.create_image(500*self.size-11, 65*self.size,image=self.photoB)
		self.canvas_bottom.addtag_withtag('image',self.pB)
	
	def del_pW(self):
		'''太极图白色区域的删除'''
		self.canvas_bottom.delete(self.pW)
	
	def del_pB(self):
		'''太极图黑色区域的删除'''
		self.canvas_bottom.delete(self.pB)
	
	def shadow(self,event):
		'''显示鼠标移动下棋子的移动'''
		if not self.stop:
			# 找到最近格点，在当前位置靠近的格点出显示棋子图片，并删除上一位置的棋子图片
			if (20*self.size<event.x<380*self.size) and (20*self.size<event.y<380*self.size):
				dx=(event.x-20*self.size)%self.dd # 鼠标x位置对边长求模的余数
				dy=(event.y-20*self.size)%self.dd # 鼠标y位置对边长求模的余数
				# 计算出预落棋位置，并创建棋子图片
				self.cross=self.canvas_bottom.create_image(event.x-dx+round(dx/self.dd)*self.dd+22*self.p, event.y-dy+round(dy/self.dd)*self.dd-27*self.p,image=self.photoWBU_list[self.present])
				# 给图片添加image标签
				self.canvas_bottom.addtag_withtag('image',self.cross)
				# 若上一个预落棋图片不为空，则将其清除
				if self.cross_last!=None:
					self.canvas_bottom.delete(self.cross_last)
				# 将当前预落棋图片设为上一个预落棋图片
				self.cross_last=self.cross
	
	def getDown(self,event):
		'''落子，并驱动玩家的轮流下棋行为'''
		if not self.stop:
			# 先找到最近格点
			if (20*self.size-self.dd*0.4<event.x<self.dd*0.4+380*self.size) and (20*self.size-self.dd*0.4<event.y<self.dd*0.4+380*self.size):
				# 鼠标x位置对边长求模的余数
				dx=(event.x-20*self.size)%self.dd
				# 鼠标y位置对边长求模的余数
				dy=(event.y-20*self.size)%self.dd
				# 求出最近格点位置
				x=int((event.x-20*self.size-dx)/self.dd+round(dx/self.dd)+1)
				y=int((event.y-20*self.size-dy)/self.dd+round(dy/self.dd)+1)
				# 判断位置是否已经被占据
				if self.positions[y][x]==0:
					# 若未被占据，则尝试占据，获得占据后能杀死的棋子列表
					self.positions[y][x]=self.present+1
					# 创建棋子落下的图片
					self.image_added=self.canvas_bottom.create_image(event.x-dx+round(dx/self.dd)*self.dd+4*self.p, event.y-dy+round(dy/self.dd)*self.dd-5*self.p,image=self.photoWBD_list[self.present])
					# 为图片添加image标签
					self.canvas_bottom.addtag_withtag('image',self.image_added)
					# 棋子与位置标签position[x,y]绑定，方便“杀死”
					self.canvas_bottom.addtag_withtag('position'+str(x)+str(y),self.image_added)
					# 获取该棋子周围棋子的存活情况（若为队友则为空；若未对手棋子，有气则为空，无气则返回【死亡棋子列表】）
					deadlist=self.get_deadlist(x,y)
					# 杀死死亡的对手棋子
					self.kill(deadlist)
					# 判断是否重复棋局（即是否存在打劫现象）
					if not self.last_2_positions==self.positions:
						# 如果没有打劫现象
						# 判断是否属于“有气”或“杀死对方”其中之一
						if len(deadlist)>0 or self.if_dead([[x,y]],self.present+1,[x,y])==False:
							# 当不重复棋局，且属于有气和杀死对方其中之一时，落下棋子有效
							# 如果可以悔棋，则将按钮激活，否则设置下一局可以悔棋
							if not self.regretchance==1:
								self.regretchance+=1
							else:
								self.regretButton['state']=NORMAL
							# 拷贝3 <- 拷贝2，拷贝3保存上上上一次棋局
							self.last_3_positions=copy.deepcopy(self.last_2_positions)
							# 拷贝2 <- 拷贝1，拷贝2保存上上一次棋局
							self.last_2_positions=copy.deepcopy(self.last_1_positions)
							# 拷贝1 <- 当前棋局，拷贝3保存上一次棋局
							self.last_1_positions=copy.deepcopy(self.positions)
							# 删除上次的image_added_sign标记，重新创建image_added_sign标记
							self.canvas_bottom.delete('image_added_sign')
							# 创建椭圆对象标记当前下棋点
							self.image_added_sign=self.canvas_bottom.create_oval(event.x-dx+round(dx/self.dd)*self.dd+0.5*self.dd, event.y-dy+round(dy/self.dd)*self.dd+0.5*self.dd,event.x-dx+round(dx/self.dd)*self.dd-0.5*self.dd, event.y-dy+round(dy/self.dd)*self.dd-0.5*self.dd,width=3,outline='#3ae')
							# 给椭圆对象添加image和image_added_sign标记
							self.canvas_bottom.addtag_withtag('image',self.image_added_sign)
							self.canvas_bottom.addtag_withtag('image_added_sign',self.image_added_sign)
							# 在界面右侧更新当前棋子颜色提示图
							if self.present==0:
								self.create_pW()
								self.del_pB()
								self.present=1
							else:
								self.create_pB()
								self.del_pW()
								self.present=0
						else:
							# 不属于杀死对方且不属于有气，则判断为无气，警告并弹出警告框
							self.positions[y][x]=0
							# 删除图片
							self.canvas_bottom.delete('position'+str(x)+str(y))
							# 提示音警告，窗口警告
							self.bell()
							self.showwarningbox('无气',"您被包围了！")
					else:
						# 重复棋局，警告打劫
						# 将该棋子删除
						self.positions[y][x]=0
						self.canvas_bottom.delete('position'+str(x)+str(y))
						# 恢复原本杀死的棋子
						self.recover(deadlist,(1 if self.present==0 else 0))
						# 提示音警告，窗口警告
						self.bell()
						self.showwarningbox("打劫","此路不通！")
				else:
					# 该位置已有棋子，声音警告
					self.bell()
			else:
				# 超出边界，声音警告
				self.bell()
	
	def if_dead(self,deadList,yourChessman,yourPosition):
		'''
		判断棋子（种类为yourChessman，位置为yourPosition）是否无气（死亡），有气则返回False，无气则返回无气棋子的列表
		本函数是游戏规则的关键，初始deadlist只包含了自己的位置，每次执行时，函数尝试寻找yourPosition周围有没有空的位置，有则结束，返回False代表有气；
		若找不到，则找自己四周的同类（不在deadlist中的）是否有气，即调用本函数，无气，则把该同类加入到deadlist，然后找下一个邻居，只要有一个有气，返回False代表有气；
		若四周没有一个有气的同类，返回deadlist,至此结束递归

		传入参数例子：[[x+i,y]],(2 if self.present==0 else 1),[x+i（列）,y（行）]
		'''
		for i in [-1,1]:
			# 若右侧、左侧位置不在死亡清单里
			if [yourPosition[0]+i,yourPosition[1]] not in deadList:
				#右侧、左侧为空，则有气
				if self.positions[yourPosition[1]][yourPosition[0]+i]==0:
					return False
			# 若下侧、上侧位置不在死亡清单里
			if [yourPosition[0],yourPosition[1]+i] not in deadList:
				#下侧、上侧为空，则有气
				if self.positions[yourPosition[1]+i][yourPosition[0]]==0:
					return False
		# 右侧不在死亡清单里面右边是你的棋子
		if ([yourPosition[0]+1,yourPosition[1]] not in deadList) and (self.positions[yourPosition[1]][yourPosition[0]+1]==yourChessman):
			# 检测右边的棋子是否有气
			midvar=self.if_dead(deadList+[[yourPosition[0]+1,yourPosition[1]]],yourChessman,[yourPosition[0]+1,yourPosition[1]])
			# 有气
			if not midvar:
				return False
			else:
				# 无气，则加入新的死亡列表
				deadList+=copy.deepcopy(midvar)
		# 左侧同上
		if ([yourPosition[0]-1,yourPosition[1]] not in deadList) and (self.positions[yourPosition[1]][yourPosition[0]-1]==yourChessman):
			midvar=self.if_dead(deadList+[[yourPosition[0]-1,yourPosition[1]]],yourChessman,[yourPosition[0]-1,yourPosition[1]])
			if not midvar:
				return False
			else:
				deadList+=copy.deepcopy(midvar)
		#下侧同上
		if ([yourPosition[0],yourPosition[1]+1] not in deadList) and (self.positions[yourPosition[1]+1][yourPosition[0]]==yourChessman):
			midvar=self.if_dead(deadList+[[yourPosition[0],yourPosition[1]+1]],yourChessman,[yourPosition[0],yourPosition[1]+1])
			if not midvar:
				return False
			else:
				deadList+=copy.deepcopy(midvar)
		# 上侧同上
		if ([yourPosition[0],yourPosition[1]-1] not in deadList) and (self.positions[yourPosition[1]-1][yourPosition[0]]==yourChessman):
			midvar=self.if_dead(deadList+[[yourPosition[0],yourPosition[1]-1]],yourChessman,[yourPosition[0],yourPosition[1]-1])
			if not midvar:
				return False
			else:
				deadList+=copy.deepcopy(midvar)
		return deadList
	
	def showwarningbox(self,title,message):
		'''警告消息框，接受标题和警告信息'''
		# 不显示预落棋位置
		self.canvas_bottom.delete(self.cross)
		# 报警
		tkinter.messagebox.showwarning(title,message)
	
	def get_deadlist(self,x,y):
		'''落子后，依次判断四周是否有棋子被杀死，并返回（对手）死棋位置列表'''
		deadlist=[]
		for i in [-1,1]:
			# 若右左是对手的棋子且没在死亡的名单里
			if self.positions[y][x+i]==(2 if self.present==0 else 1) and ([x+i,y] not in deadlist):
				# 返回右左对手棋子的存活情况（有气或无气返回【死亡棋子列表】）
				killList=self.if_dead([[x+i,y]],(2 if self.present==0 else 1),[x+i,y])
				# 若无气则加入【对手死亡列表】
				if not killList==False:
					deadlist+=copy.deepcopy(killList)
			# 若下上是对手的棋子且没在死亡的名单里，操作同上
			if self.positions[y+i][x]==(2 if self.present==0 else 1) and ([x,y+i] not in deadlist):		
				killList=self.if_dead([[x,y+i]],(2 if self.present==0 else 1),[x,y+i])
				if not killList==False:
					deadlist+=copy.deepcopy(killList)
		return deadlist
	
	def recover(self,list_to_recover,b_or_w):
		'''恢复位置列表list_to_recover为b_or_w指定的棋子'''
		if len(list_to_recover)>0:
			for i in range(len(list_to_recover)):
				self.positions[list_to_recover[i][1]][list_to_recover[i][0]]=b_or_w+1
				# 添加落下棋子的图片
				self.image_added=self.canvas_bottom.create_image(20*self.size+(list_to_recover[i][0]-1)*self.dd+4*self.p, 20*self.size+(list_to_recover[i][1]-1)*self.dd-5*self.p,image=self.photoWBD_list[b_or_w])
				# 为落下棋子图片添加image标签
				self.canvas_bottom.addtag_withtag('image',self.image_added)
				# 为落下棋子图片添加position[x,y]（列，行）标签
				self.canvas_bottom.addtag_withtag('position'+str(list_to_recover[i][0])+str(list_to_recover[i][1]),self.image_added)
	
	def kill(self,killList):
		'''杀死位置列表killList中的棋子，即删除图片，位置值置0'''
		if len(killList)>0:
			for i in range(len(killList)):
				# 棋子置数为0
				self.positions[killList[i][1]][killList[i][0]]=0
				# 删除图片
				self.canvas_bottom.delete('position'+str(killList[i][0])+str(killList[i][1]))
	
	def keyboardQuit(self,event):
		'''键盘快捷键退出游戏'''
		self.quit()
	
	def newGame1(self):
		'''修改全局变量值，newApp使主函数循环，以建立不同参数的对象'''
		global mode_num,newApp
		# 9路改成13路
		mode_num=(13 if self.mode_num==9 else 9)
		newApp=True
		self.quit()
	
	def newGame2(self):
		'''修改全局变量值，newApp使主函数循环，以建立不同参数的对象'''
		global mode_num,newApp
		#19路改成13路
		mode_num=(13 if self.mode_num==19 else 19)
		newApp=True
		self.quit()

# 声明全局变量，用于新建Application对象时切换成不同模式的游戏
global mode_num,newApp
# 棋的路数
mode_num=9
# 是否重新开一局
newApp=False

if __name__=='__main__':
	# 循环，直到不切换游戏模式
	while True:
		newApp=False
		app=Application(mode_num)
		app.title('围棋')
		app.mainloop()
		if newApp:
			app.destroy()
		else:
			break