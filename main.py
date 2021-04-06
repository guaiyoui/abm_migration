from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import math
import random
from tqdm import tqdm
import cv2
import stats

def compute_Wm():
    return alpha * A_m * math.pow(N_m, alpha-1)

def compute_Wa():
    Y_a = A_a * math.pow(N_a, phi)
    # print(Y_a)
    Y_m = A_m * math.pow(N_m, alpha)
    P = rho * math.pow(Y_m/Y_a, sigma)
    return phi * A_a * P * math.pow(N_a, phi-1)

def compute_Nm_opt():
    return math.pow(alpha*A_m/W_m, 1/(1-alpha))

#  0: 留在农村， 1: 去城里没有工资  2:  去城里有工资
def init_list(width, height):
    matrix = np.zeros((height, width))
    migra_list = np.zeros((height, width))
    total = width*height
    to_urban = int(total*N_mu)
    index = random.sample(range(0, total), to_urban)
    for i in range(to_urban):
        row = int(index[i]/width)
        col = index[i]%width
        matrix[row][col] = 2
    return matrix, migra_list


def plot_list(data, title):
    # tips = sns.load_dataset("tips")
    # sns.jointplot(x = "total_bill", y = "tip", data = tips, kind ="hex", color="lightcoral")
    # with sns.axes_style("dark"):
    #     sns.jointplot(x, y, kind="hex")

    plt.title("simulation steps: "+title)
    plt.imshow(data, cmap='Blues')
    # plt.colorbar()
    plt.savefig("./image/"+title+".png")

def plot_line(data, title):
    y = data
    x = a = np.arange(0,len(data),1)
    plt.figure(figsize=(8,4)) #创建绘图对象
    plt.plot(x,y,"b--",linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("simulation steps") #X轴标签
    plt.ylabel(title)  #Y轴标签
    plt.title(title) #图标题
    plt.savefig(title+'.png')

"""A model with some number of agents."""
class MigrationModel(Model):
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, False)  # True意味着边界是循环的，从一端出去，将从二维平面相反边进入, 这里用False
        self.schedule = RandomActivation(self)   # 串行随机调用
        self.running = True
    
        for i in range(self.num_agents):
            a = RuralAgent(i, self)
            self.schedule.add(a)
            x = int(i/self.grid.width)
            y = i%self.grid.width
            self.grid.place_agent(a, (x, y))

    def update_data(self):
        global migra_list
        global origin_list
        global N_a
        global N_m
        global N_mu
        global Nm_opt
        global N_mu_share
        global unemploy_rate
        # print("--------------------")
        N_mu = np.sum(migra_list)/(2*self.grid.width*self.grid.height)
        N_mu_share.append(N_mu)
        # print(N_mu)
        # print(Nm_opt)
        if N_mu <= Nm_opt:
            origin_list = migra_list
            N_a = 1-N_mu
            N_m = N_mu
            migra_list = np.zeros((self.grid.height, self.grid.width))
        else:
            N_a = 1-N_mu
            N_m = Nm_opt

            total_migr = np.sum(migra_list)/2
            over_ = int(total_migr - self.grid.width*self.grid.height*Nm_opt)
            index = random.sample(range(0, int(total_migr)), over_)
            # print(index)
            count = 0
            for i in range(self.grid.height):
                for j in range(self.grid.width):
                    if migra_list[i][j] != 0:
                        if count in index:
                            migra_list[i][j] = 1
                        count += 1
            origin_list = migra_list
            migra_list = np.zeros((self.grid.height, self.grid.width))
        
        # print(1-N_m/N_mu)
        unemploy_rate.append(1-N_m/N_mu)
        # print(migra_list)
        # print(origin_list)

        # print(N_a)
        # print(N_m)
        # print(compute_Wa())
        # print(compute_Wm())

        # 更新salary
        for i in range(self.grid.height):
            for j in range(self.grid.width):
                item = (i, j)
                cellmates = self.grid.get_cell_list_contents(item)
                if origin_list[i][j] == 2:
                    cellmates[0].salary = compute_Wm()
                elif origin_list[i][j] == 1:
                    cellmates[0].salary = 0
                else:
                    cellmates[0].salary = compute_Wa()
        
    def step(self, i):
        # self.datacollector.collect(self)
        self.schedule.step()
        self.update_data()
        global origin_list
        global migra_list
        # print(migra_list)
        plot_list(origin_list, str(i))

class RuralAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        global origin_list
        # print(model.grid.width)
        x = int(unique_id/model.grid.width)
        y = unique_id%model.grid.width
        # self.salary = 
        # print(origin_list)
        # print(self.pos)
        if origin_list[x, y] == 0:
            self.salary = compute_Wa()
        else:
            self.salary = compute_Wm()
        # print(self.salary)
    
    def update_salary(self):
        global origin_list
        global migra_list

        is_possible_mig = random.random()
        # print(is_possible_mig)
        if origin_list[self.pos[0], self.pos[1]] == 1:
            # self.salary = compute_Wa()
            migra_list[self.pos[0], self.pos[1]] = 0
        elif origin_list[self.pos[0], self.pos[1]] == 0:
            if is_possible_mig > 0.75:
                possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=False,   # moore包含周边八个结构，如果使用Neumann只包含正交的四个位置
                include_center=False) # 不包含自己
                
                total_salary = 0
                for item in possible_steps:
                    cellmates = self.model.grid.get_cell_list_contents(item)
                    total_salary += cellmates[0].salary
                if total_salary/len(possible_steps) > self.salary:
                    migra_list[self.pos[0], self.pos[1]] = 2
        else:
            migra_list[self.pos[0], self.pos[1]] = 2


    def step(self):
        # self.move()
        self.update_salary()

def gen_video(length):
    # 图片的名字
    name = ['origin']
    for i in range(length):
        name.append(str(i+1))
    # name = ['1','2','3','4','5','6','7','8','9','10','11','12']
    # 设置video的fps，位置等等，初始化
    video=cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc(*'MJPG'),2,(1280,720))  #定义保存视频目录名称及压缩格式，fps=10,像素为1280*720
    # 循环将图片写入视频
    for i in range(0,length):
        img=cv2.imread('./image/'+name[i]+'.png')  #读取图片
        img=cv2.resize(img,(1280,720)) #将图片转换为1280*720
        video.write(img)


if __name__ == '__main__': 

    A_a = 1
    A_m = 1
    phi = 0.3
    alpha = 0.7
    sigma = 1
    rho = 1
    W_m = 0.8
    N_mu = 0.2
    N_a = 1 - N_mu
    N_m = 0.2
    N_mu_share = [N_mu]
    unemploy_rate = [1-N_m/N_mu]
    Nm_opt = compute_Nm_opt()

    width = 300
    height = 300
    iterations = 40

    origin_list, migra_list = init_list(height,width)
    plot_list(origin_list, 'origin')
    # model = MigrationModel(height*width, width, height)
    # for i in tqdm(range(iterations)):
    #     model.step(i)
    unemploy_rate = [0.0, 0.0, 0.0, 0.0, 0.011549998082032276, 0.10983962071039044, 0.07073785794098952, 0.08793335212408948, 0.08327928510502691, 0.08422552861751142, 0.08301690260779993, 0.08349782251202975, 0.08243329230539742, 0.08650303328267406, 0.08243329230539742, 0.08301690260779993, 0.08227266921451881, 0.08538745778249579, 0.08240409228924106, 0.08434185436808406, 0.08440000616191301, 0.08416735465795211, 0.08474876187312608, 0.08359977098161309, 0.08343955605871023, 0.0849085194640018, 0.0850972520006017, 0.0849085194640018, 0.083220990860562, 0.084283695187086, 0.08364345624009928, 0.0849811181153518, 0.08268141742916568, 0.08349782251202975, 0.08564848562076954, 0.08257926451426101, 0.0861121674131079, 0.08260845338289136, 0.08548898631602042, 0.08165886343241491, 0.08453082069598095]

    # plot_line(N_mu_share, "urban share")
    plot_line(unemploy_rate, "unemployment_rate")
    gen_video(iterations)
    print(N_mu_share)
    print(unemploy_rate)