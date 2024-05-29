'''
@author: shaw wang
@email: shawwang@yeah.net
'''

import io
import os
import json
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class DataGen:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.json_dir = os.path.join(self.data_dir, 'json')
        self.img_dir = os.path.join(self.data_dir, 'img')
        self.data_list = []
    
    def run(self):
        for filename in os.listdir(self.json_dir):
            if os.path.isfile(os.path.join(self.json_dir, filename)):
                self.data_list.append(filename)
        # print(self.data_list)
        for filename in self.data_list:
            self.get_input(filename)
    
    def get_input(self, filename):
        with open(os.path.join(self.json_dir, filename), 'r') as f:
            data = json.load(f)

        vertices = np.array(data['vertices'])
        # 设置一个固定的图像大小
        plt.figure(figsize=(4, 4))  
        # 设置x轴和y轴的范围为（-10, 10）
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        # 绘制图形，但暂时不关闭坐标轴，以便设置范围
        plt.plot(vertices[:, 0], vertices[:, 1], 'k-')
        plt.arrow(0, 0, 0, 1, length_includes_head=True, head_width=0.3, fc='b', ec='g')
        plt.arrow(data['target'][0], data['target'][1], np.cos(data['target'][2]), np.sin(data['target'][2]), 
                  length_includes_head=True, head_width=0.3, fc='b', ec='r')
        plt.axis("equal")
        # 现在关闭坐标轴
        plt.axis('off')
        # 使用 BytesIO 缓冲区保存绘制的图像
        buf = io.BytesIO()
        plt.savefig(buf, dpi=100, pad_inches=0, format='png')
        # 从缓冲区读取图像并转换为 PIL 图像
        buf.seek(0)
        pil_image = Image.open(buf).convert('L')  # 转换为灰度图像
        buf.close()
        pil_image.save(os.path.join(self.img_dir, 'input', filename[:-5] + '.png'))

        c = np.array(data['traj_coeff'])
        T = np.array(data['traj_t'])
        px, py = self._poly_curve_5(c, T)
        plt.plot(px, py)
        # 使用 BytesIO 缓冲区保存绘制的图像
        buf = io.BytesIO()
        plt.savefig(buf, dpi=100, pad_inches=0, format='png')
        # 从缓冲区读取图像并转换为 PIL 图像
        buf.seek(0)
        pil_image = Image.open(buf).convert('L')  # 转换为灰度图像
        buf.close()
        pil_image.save(os.path.join(self.img_dir, 'output', filename[:-5] + '.png'))

        plt.close()


    def _poly_curve_5(self, c, T):
        t = np.linspace(0, T, 1000)
        px = c[0,0]*t**5+c[0,1]*t**4+c[0,2]*t**3+c[0,3]*t**2+c[0,4]*t+c[0,5]
        py = c[1,0]*t**5+c[1,1]*t**4+c[1,2]*t**3+c[1,3]*t**2+c[1,4]*t+c[1,5]
        return px, py

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    data_gen = DataGen(data_dir)
    data_gen.run()