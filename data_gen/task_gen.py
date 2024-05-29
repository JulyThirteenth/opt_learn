'''
@author: shaw wang
@email: shawwang@yeah.net
'''

import os
import json
import time
import numpy as np
import nums_from_string as nfs
import matplotlib
import matplotlib.pyplot as plt  
from matplotlib.widgets import TextBox, Button
from pyomo.environ import *
from pyomo.dae import *

matplotlib.use('Qt5Agg')

def is_convex(vertices): 
    vertices = np.array(vertices)
    # 确保至少有三个顶点  
    if len(vertices) < 3:  
        return False     
    # 计算向量叉积的函数  
    def cross_product(v1, v2):  
        return v1[0] * v2[1] - v1[1] * v2[0]     
    # 遍历多边形的边并检查叉积的符号  
    dir = cross_product(vertices[1] - vertices[0], vertices[2] - vertices[1]) > 0  
    for i in range(2, len(vertices)-1):  
        cp = cross_product(vertices[i] - vertices[i-1], vertices[i+1] - vertices[i])  
        if (cp > 0 and not dir) or (cp < 0 and dir):  
            # 如果叉积的符号与之前的不同，则多边形是凹的  
            return False  
    # 如果所有叉积的符号都相同，则多边形是凸的  
    return True  
  

def polygon_orientation(vertices):
    area = 0
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        area += (vertices[j][0] + vertices[i][0]) * (vertices[j][1] - vertices[i][1])
    if area > 0:
        return "逆时针"
    elif area < 0:
        return "顺时针"
    else:
        raise Exception("Error in vertices of polygon")

def polygon_hrep(vertices):
    n = len(vertices)-1
    mat = np.zeros((n, 3))
    for i in range(n):
        v1 = vertices[i] # vertex 1
        v2 = vertices[i+1] # vertex 2
        if v1[0] == v2[0]: # perpendicular hyperplane
            if v1[1] < v2[1]:
                mat[i][:2] = [1, 0]
                mat[i][2] = v1[0]
            else:
                mat[i][:2] = [-1, 0]
                mat[i][2] = -v1[0]
        elif v1[1] == v2[1]: # horizontal hyperplane
            if v1[0] < v2[0]:
                mat[i][:2] = [0, -1]
                mat[i][2] = -v1[1]
            else:
                mat[i][:2] = [0, 1]
                mat[i][2] = v1[1]
        else: # general formula
            a = (v1[1]-v2[1])/(v1[0]-v2[0]) # k
            b = (v1[0]*v2[1]-v2[0]*v1[1])/(v1[0]-v2[0]) # b
            if (v1[1] < v2[1] and a < 0) or (v1[1] > v2[1] and a > 0):
                mat[i] = [-a, 1, b]
            else:
                mat[i] = [a, -1, -b]  
    return mat

N = 30
v_max = 1.0
k_max = 2.0

class TaskGen:
    def __init__(self):
        from traj_opt import TrajOpt
        self.traj_opt = TrajOpt(N, v_max, k_max) # optimization
        self.target_x = None # record end x corrdinate
        self.target_y = None # record end y corrdinate
        self.target_t = None # record end t corrdinate
        self.vertices = []
        self.vertices_mode = True # True for vertices, False for target
        self.px = None # record opti traj corrd x
        self.py = None # record opti traj corrd y
        self.fig, self.ax = plt.subplots(num="Task Generation")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        # 连接事件处理器
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)  
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        # 鼠标状态
        self.pressed = False
        self.pressed_x = None
        self.pressed_y = None
        # 导航栏状态
        self.navi_act = False
        # 添加文本输入框
        self.text_box = TextBox(plt.axes([0.3, 0.01, 0.4, 0.05]), 'input', color='gray', hovercolor='white')
        self.text_box.on_submit(self.on_submit)
        # 添加删除顶点按钮
        self.del_button = Button(plt.axes([0.0, 0.9, 0.2, 0.05]), 'delete vertex', color='gray', hovercolor='blue')
        # 添加设置模式按钮
        self.mode_button = Button(plt.axes([0.2, 0.9, 0.2, 0.05]), 'mode', color='gray', hovercolor='blue')
        # 添加保存数据按钮
        self.save_button = Button(plt.axes([0.4, 0.9, 0.2, 0.05]), 'save', color='gray', hovercolor='blue')
        # 添加清空数据按钮
        self.clear_button = Button(plt.axes([0.6, 0.9, 0.2, 0.05]), 'clear', color='gray', hovercolor='blue')
        self.ax.set_xlim([-10, 10]) # 设置轴的放缩状态
        self.ax.set_ylim([-10, 10]) # 设置轴的放缩状态
        plt.show()

    def draw(self):
        # 获取当前轴的放缩状态
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.ax.clear()  # 清空之前的图形

        # 绘制目标
        if self.target_x is not None and self.target_y is not None and self.target_t is not None:
            self.ax.arrow(self.target_x, self.target_y, 1.0*np.cos(self.target_t), 1.0*np.sin(self.target_t), head_width=0.15, head_length=0.3, fc='b', ec='b')

        # 绘制顶点
        self.ax.scatter([p[0] for p in self.vertices], [p[1] for p in self.vertices], color='red')
        # 绘制连接线段
        if len(self.vertices) > 1:
            for i in range(1, len(self.vertices)):
                self.ax.plot([self.vertices[i-1][0], self.vertices[i][0]], [self.vertices[i-1][1], self.vertices[i][1]], color='blue')
            # 绘制封闭图形
            if len(self.vertices) > 2:
                self.ax.plot([self.vertices[-1][0], self.vertices[0][0]], [self.vertices[-1][1], self.vertices[0][1]], color='blue')
        
        # 绘制轨迹
        if self.px is not None and self.py is not None:
            self.ax.plot(self.px, self.py)

        # 设置轴的放缩状态
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        # 更新图形
        self.fig.canvas.draw()

    def onclick(self, event):
        self.pressed = True
        self.pressed_x = event.xdata
        self.pressed_y = event.ydata

    def onrelease(self, event):
        self.pressed = False
        if self.ax.get_navigate_mode() in ["PAN", "ZOOM"]: # 如果导航栏处于活动状态则不做任何处理
            self.navi_act = True
            return
        else:
            self.navi_act = False
        if event.button == 1: # 检测鼠标左键释放事件
            if event.inaxes == self.save_button.ax: # 检查是否点击了保存按钮
                self.on_save_button_clicked(event) # 调用保存函数
            elif event.inaxes == self.del_button.ax: # 检查是否点击了删除按钮
                self.on_del_button_clicked(event) # 调用删除函数
            elif event.inaxes == self.mode_button.ax: # 检查是否点击了模式按钮
                self.vertices_mode = not self.vertices_mode
            elif event.inaxes == self.ax and self.vertices_mode: # 检查是否点击了添加点
                self.vertices.append([event.xdata, event.ydata]) # 添加点到列表中
                self.draw()
            elif event.inaxes == self.clear_button.ax: # 检查是否点击了清空按钮
                self.px = None
                self.py = None
                self.vertices.clear()
                self.target_t = None
                self.target_x = None
                self.target_y = None
                self.draw()


    def onmotion(self, event):
        if event.inaxes == self.ax and self.pressed and not self.vertices_mode and not self.navi_act:
            self.set_target = True
            dx = event.xdata - self.pressed_x
            dy = event.ydata - self.pressed_y
            dt = np.arctan2(dy, dx)
            self.target_t = dt
            self.target_x = self.pressed_x
            self.target_y = self.pressed_y
            # print(self.end_x, self.end_y, self.end_t, sep=' ')
        # 更新图形
        self.draw()
            
    # 定义一个回调函数来处理输入框中的文本
    def on_submit(self, text):
        self.text_box.set_val("")
        corrd = nfs.get_nums(text)
        if corrd.__len__() == 2:
            self.vertices.append([corrd[0], corrd[1]])
        self.draw() 

    # 删除按钮回调函数
    def on_del_button_clicked(self, event):
        if self.vertices:
            self.vertices.pop()
            self.draw() 
    
    # 保存按钮回调函数
    def on_save_button_clicked(self, event):
        if self.vertices != [] and self.target_t != None and self.target_x != None and self.target_y != None:
            # it is assumed that the vertices are given in CLOCK-WISE, 
            # and that the first vertex is repeated at the end of the vertex list
            self.vertices.append(self.vertices[0]) 
            if is_convex(self.vertices):
                if polygon_orientation(self.vertices) == "顺时针": # 保证逆时针排列
                    self.vertices.reverse()
                task = dict()
                task['vertices'] = self.vertices
                task['hrep'] = polygon_hrep(self.vertices).tolist()
                task['target'] = [self.target_x, self.target_y, self.target_t, 1.0]
                self.traj_opt.set_target(*task['target'])
                self.traj_opt.set_freespace(task['hrep'])
                print("opt solving...")
                Tg, c = self.traj_opt.solve()
                print("opt solved!")
                def PolyCurve5(c, T):
                    t = np.linspace(0, T, 1000)
                    px = c[0,0]*t**5+c[0,1]*t**4+c[0,2]*t**3+c[0,3]*t**2+c[0,4]*t+c[0,5]
                    py = c[1,0]*t**5+c[1,1]*t**4+c[1,2]*t**3+c[1,3]*t**2+c[1,4]*t+c[1,5]
                    return px, py
                self.px, self.py = PolyCurve5(c, Tg)
                task['traj_t'] = Tg
                task['traj_coeff'] = c.tolist()
                file_dir = os.path.join(os.path.dirname(__file__), 'data/json/')
                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                file_name = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
                file_path = os.path.join(file_dir, file_name + ".json")
                with open(file_path, 'w') as f:
                    json.dump(task, f, indent=4, ensure_ascii=False)
                f.close()
                # file_path = os.path.join(file_dir,"hrep.npy")
                # np.save(file_path, polygon_hrep(self.vertices)) # self.points逆时针排列
                # file_path = os.path.join(file_dir,"vertices.csv")
                # with open(file_path, "w") as f:
                #     f.write("x, y\n")
                #     for point in self.vertices:
                #         f.write(f"{point[0]:.6f}, {point[1]:.6f}" + "\n")
        self.draw()

if __name__ == "__main__":
    TaskGen()
