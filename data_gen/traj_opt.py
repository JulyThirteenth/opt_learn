'''
@author: shaw wang
@email: shawwang@yeah.net
'''

from pyomo.environ import *
from pyomo.dae import *
import numpy as np

class TrajOpt:
    def __init__(self, N, v_max, k_max) -> None:
        self.N = N
        self.v_max = v_max
        self.k_max = k_max
        self.target_x = None
        self.target_y = None
        self.target_t = None
        self.target_v = None
        self.A = None
    
    def set_target(self, x, y, t, v):
        self.target_x = x
        self.target_y = y
        self.target_t = t
        self.target_v = v
    
    def set_freespace(self, A):
        self.A = A

    def solve(self):
        m = ConcreteModel()

        # Parameters
        pw = {(0, 0):1.0, (0, 1):1.0, (0, 2):1.0, (0, 3):1.0, (0, 4):1.0, (0, 5):1.0,
            (1, 0):0.0, (1, 1):5.0, (1, 2):4.0, (1, 3):3.0, (1, 4):2.0, (1, 5):1.0,
            (2, 0):0.0, (2, 1):0.0, (2, 2):20.0,(2, 3):12.0,(2, 4):6.0, (2, 5):2.0,
            (3, 0):0.0, (3, 1):0.0, (3, 2):0.0, (3, 3):60.0,(3, 4):24.0,(3, 5):6.0}

        m.pw = Param(RangeSet(0, 3), RangeSet(0, 5), initialize=pw)

        # 多项式系数
        '''
        x = a0 + a1*t^1 + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        y = b0 + b1*t^1 + b2*t^2 + b3*t^3 + b4*t^4 + b5*t^5
        '''
        m.c = Var(RangeSet(0, 1), RangeSet(0, 5))

        # 多项式曲线采样点的0-3阶导数
        '''
        [
            # oh # pos[x, y] # [[a1,...],[a2,...]]
            # 1st # vel[x, y] # [[a1,...],[a2,...]]
            # 2nd # acc[x, y] # [[a1,...],[a2,...]]
            # 3rd # jerk[x, y] # [[a1,...],[a2,...]]
        ]
        '''
        m.p = Var(RangeSet(0, 3), RangeSet(0, 1), RangeSet(0, self.N-1))

        # 时间
        m.dt = Var(domain=NonNegativeReals)

        # 多项式曲线公式约束
        m.poly_cons = Constraint(RangeSet(0, 3), RangeSet(0, 1), RangeSet(0, self.N-1), rule=self._polyfun)

        # 速度约束
        m.speed_con = Constraint(RangeSet(0, self.N-1), rule=self._velfun)

        # 起点约束 位置、姿态、速度、曲率
        m.conslist = ConstraintList()
        m.conslist.add(expr=(m.p[0, 0, 0] == 0)) # x
        m.conslist.add(expr=(m.p[0, 1, 0] == 0)) # y
        m.conslist.add(expr=(m.p[1, 0, 0] == 0)) # theta v_x
        m.conslist.add(expr=(m.p[1, 1, 0] == 0)) # theta v_y

        # 目标约束 位置、姿态、速度、曲率
        m.conslist.add(expr=m.p[0, 0, self.N-1] == self.target_x) # x
        m.conslist.add(expr=m.p[0, 1, self.N-1] == self.target_y) # y
        m.conslist.add(expr=m.p[1, 0, self.N-1] == self.target_v*cos(self.target_t)) # theta v
        m.conslist.add(expr=m.p[1, 1, self.N-1] == self.target_v*sin(self.target_t)) # theta v

        # 曲率约束
        m.kappa_con = Constraint(RangeSet(0, self.N-1), rule=self._kappafun)

        # 自由空间约束
        m.freesapce_con = Constraint(RangeSet(0, self.N-1), RangeSet(0, len(self.A)-1), rule=self._freespacefun)

        # 目标函数中各指标项的权值
        m.coeffs = Param(RangeSet(0, 1), initialize={0:10.0, 1:0.2}, mutable=True)
        # 目标函数
        m.uxobj = m.coeffs[0]*sum(m.p[3, 0, k]**2 for k in RangeSet(0, self.N-1))
        m.uyobj = m.coeffs[0]*sum(m.p[3, 1, k]**2 for k in RangeSet(0, self.N-1))
        m.tobj = m.coeffs[1]*m.dt*self.N
        m.obj = Objective(expr=m.uxobj+m.uyobj+m.tobj, sense=minimize)
        SolverFactory('ipopt').solve(m)
        m.display()

        Tg = m.dt()*(self.N-1)
        c = np.array([[m.c[i,j]() for j in range(6)] for i in range(2)])
        return Tg, c

    def _polyfun(self, m, k, i, j):
        '''
        k: 第k阶导数
        i: 0 for x, 1 for y
        j: 第j个路点
        '''
        return (m.p[k, i, j] == m.pw[k, 0]*m.c[i, max(0-k, 0)]*(m.dt*j)**5+
                                m.pw[k, 1]*m.c[i, max(1-k, 0)]*(m.dt*j)**4+
                                m.pw[k, 2]*m.c[i, max(2-k, 0)]*(m.dt*j)**3+
                                m.pw[k, 3]*m.c[i, max(3-k, 0)]*(m.dt*j)**2+
                                m.pw[k, 4]*m.c[i, max(4-k, 0)]*(m.dt*j)**1+
                                m.pw[k, 5]*m.c[i, max(5-k, 0)])    
    
    def _velfun(self, m, i):
        '''
        i: 第i个路点
        '''
        return (m.p[1, 0, i]**2 + m.p[1, 1, i]**2 <= self.v_max*self.v_max)

    def _kappafun(self, m, i):
        '''
        i: 第i个路点
        kappa = (v_x*a_y - v_y*a_x)**2/(v_x**2+v_y**2)**(3/2)
        '''
        return (m.p[2,1,i]*m.p[1,0,i]-m.p[2,0,i]*m.p[1,1,i])**2<=self.k_max**2*(m.p[1,0,i]**2+m.p[1,1,i]**2)**(3)
    
    def _freespacefun(self, m, i, j):
        '''
        i: 第i个路点
        j: 第j个自由空间半平面约束
        '''
        return m.p[0, 0, i]*self.A[j][0] + m.p[0, 1, i]*self.A[j][1] <= self.A[j][2] 
    

if __name__ == '__main__':
    
    import os, json
    import numpy as np
    import matplotlib.pyplot as plt

    N = 30
    v_max = 1.0
    k_max = 1.0
    traj_opt = TrajOpt(N, v_max, k_max)

    with open(os.path.join(os.path.dirname(__file__), "task.json")) as f:
        task = json.load(f)
        traj_opt.set_target(*task['target'])
        traj_opt.set_freespace(task['hrep'])
        Tg, c = traj_opt.solve()

        def PolyCurve5(c, T):
            t = np.linspace(0, T, 1000)
            px = c[0,0]*t**5+c[0,1]*t**4+c[0,2]*t**3+c[0,3]*t**2+c[0,4]*t+c[0,5]
            py = c[1,0]*t**5+c[1,1]*t**4+c[1,2]*t**3+c[1,3]*t**2+c[1,4]*t+c[1,5]
            return px, py
        print(Tg, c)
        px, py = PolyCurve5(c, Tg)

        plt.plot(px, py)
        vertices = np.array(task['vertices']).T
        plt.plot(vertices[0], vertices[1])
        plt.scatter(vertices[0], vertices[1])
        plt.arrow(0, 0, 0, 1, length_includes_head=True, head_width=0.3, fc='b', ec='g')
        plt.arrow(task['target'][0], task['target'][1], np.cos(task['target'][2]), np.sin(task['target'][2]), length_includes_head=True, head_width=0.3, fc='b', ec='r')
        plt.axis("equal")
        plt.show()

    f.close()

    