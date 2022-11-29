#导入需要的库
import tkinter as tk
import numpy as np
import scipy.constants as C
from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import display
from scipy.integrate import odeint
import math


# alpha粒子质量和原子序数
m_alpha = C.physical_constants["alpha particle mass energy equivalent in MeV"][0]
m_alpha_kg = C.physical_constants["alpha particle mass"][0]
Z_alpha=2
Z_Au=79
Z_Ag=47
# 常系数：
# k1=e^2/4\pi\varepsilon_0
# k2=Z_\alpha Z_Au/m_\alpha
MeV = 1e6 * C.e
k1 = 1.44  # fm.Mev
k2 =Z_alpha * Z_Au / m_alpha * (C.c * 1e15)**2  # fm^2/(s^2 Mev)
k3 =Z_alpha * Z_Ag / m_alpha * (C.c * 1e15)**2  # fm^2/(s^2 Mev)

# 求解所用微分方程 四列分别为粒子的x正方向速度，x正方向加速度，y正方向速度和y正方向加速度
def differential_equations(y,t):
    return np.array([y[1],k1*k2*y[0] /(np.sqrt((y[0])**2+(y[2])**2))**3,y[3],k1*k2*y[2]/(np.sqrt((y[0])**2+(y[2])**2))**3])

# 由初始条件和微分方程求解出不同时刻t的坐标(x,y)的函数
def solving(val,b):    
    initial_value=np.array([-3e3,val,b,0])
    t = np.arange(0,1e-18,(1e-20))
    res = odeint(differential_equations,initial_value, t)
    x,y = res[:-1,0],res[:-1,2]
    return x,y

def solving_speed(val,b):   #返回运动各点的速度 
    initial_value=np.array([-3e3,val,b,0])
    t = np.arange(0,1e-18,(1e-20))
    res = odeint(differential_equations,initial_value, t)
    v_x,v_y = res[:-1,1],res[:-1,3]#x和y方向的速度
    return v_x,v_y

#创建颜色列表，方便后续绘图时有不同颜色可以选用
colorlist1=['skyblue','cyan','palegreen','red','hotpink','gold','brown']

# 创建所需要大小的矩阵，行数为同一入射能量求解出的数据点数，列数为不同初值设置的组数
x_m=np.ones((100,6), dtype='float')#m为手动输入绘图所需矩阵
y_m=np.ones((100,6), dtype='float')
x_r=np.ones((100,10000), dtype='float')#r为随机生成绘图所需矩阵
y_r=np.ones((100,10000), dtype='float')

v_x=np.ones((100,10000), dtype='float')#r为随机生成绘图所需矩阵
v_y=np.ones((100,10000), dtype='float')
tan_theta=np.ones((1,10000), dtype='float')


#gui部分
root=tk.Tk()#显示一个窗口
root.geometry("500x250")#设置窗口初始大小
root.title('alphal粒子散射模拟')#设置窗口标题

#创建三个空的字典，设置组建文字，并将入射能量和瞄准距离初始值设置为0
energy={}
aimdistance={}
particlerank={}
for i in range(0,6):
    j=1+i
    s1=StringVar()
    s1.set(0)
    s2=StringVar()
    s2.set(0)
    energy['E'+str(i)]=Entry(root,width=20,textvariable=s1)
    aimdistance['A'+str(i)]=Entry(root,width=20,textvariable=s2)
    particlerank['R'+str(i)]=Label(root,text='入射粒子%d'% j)

S1 = Scale(root,from_=10,to=60,tickinterval=6,resolution=10,length=120)

#手动输入绘图按钮绑定函数
def draw(event):
    for i in range(0,6):#对不同的入射初始条件循环
        Energy=float(energy['E'+str(i)].get())*MeV#获取用户手动输入入射能量数值
        v0 = np.sqrt(2 * Energy / m_alpha_kg) * 1e15#根据入射能量计算入射初速度
        b=float(aimdistance['A'+str(i)].get())#获取用户手动输入瞄准距离数值
        x_m[:,i],y_m[:,i]=solving(v0,b)#利用前边定义的求解函数求解第i组初始条件对应的坐标并将数据导入矩阵的第i列备用
    # 绘图
    for j in range(len(x_m)):#对矩阵的行数循环，先行后列的循环顺序使得图片上出现的是同一时间不同的入射初始条件在对应时间的图像
        plt.cla()# 清空图表
        plt.plot(0,0,'ko',ms=10) # 靶粒子位置
        plt.text(2,-2,'target particle',fontsize=20)#标记靶粒子位置
        plt.xlim((-3e3,3e3))#横坐标轴范围设置
        plt.ylim((-3e3,3e3))#纵坐标轴范围设置
        plt.title('alpha particle scattering')#图像标题
        plt.xlabel("x axis/fm")#横坐标轴标题
        plt.ylabel("y axis/fm")#纵坐标轴标题
        for k in range(0,6):#对矩阵的列数循环
            plt.plot(x_m[:j+1,k], y_m[:j+1,k], color=colorlist1[k],linewidth=1)# 绘制曲线，每次从第一行数据画到第j行，粒子经过的轨迹会被保留
            plt.scatter(x_m[j,k], y_m[j,k], color='red', s=15)# 绘制最新点
        display.clear_output(wait=True)
        plt.pause(1/S1.get())
    plt.show()

#随机生成绘图按钮绑定函数
def random_draw(event):
    theta_hist=[]
    missing_list=[]
    for i in range(0,10000):
        Energy=float(np.random.rand()*10+1)*MeV#随机生成不同的入射能量
        v0 = np.sqrt(2 * Energy / m_alpha_kg) * 1e15
        b=float(np.random.rand()*2000-1000)#随机生成不同的瞄准距离
        x_r[:,i],y_r[:,i]=solving(v0,b)#利用前边定义的求解函数求解第i组初始条件对应的坐标并将数据导入矩阵的第i列备用
        #v_x[:,i],v_y[:,i]=solving_speed(v0,b)#利用前边定义的求解函数求解第i组初始条件对应的速度·并将数据导入矩阵的第i列备用
        #tan_theta[:,i]=(v_y[:,i]/v_x[:,i])
    # 绘图
    figure,axes=plt.subplots(nrows=1,ncols=2,figsize=(18,9))
    for j in range(len(x_m)):#与手动输入绘图原理相同
        plt.subplot(1,2,1)
        plt.cla()# 清空图表
        plt.plot(0,0,'ko',ms=10) # 靶粒子位置
        plt.text(2,-2,'target particle',fontsize=20)
        plt.xlim((-3e3,3e3))
        plt.ylim((-3e3,3e3))
        plt.title('alpha particle scattering')
        plt.xlabel("x axis/fm")
        plt.ylabel("y axis/fm")
        #plt.plot(x_r[:j+1,:], y_r[:j+1,:],'--', color='orange',alpha=0.1,linewidth=1)# 绘制曲线，每次从第一行数据画到第j行，粒子经过的轨迹会被保留
        plt.scatter(x_r[j,:], y_r[j,:], color='red', s=15)# 绘制最新点
        display.clear_output(wait=True)
        plt.pause(0.1)
        plt.subplot(1,2,2)
        plt.cla()
        plt.xlim((0,180))#横坐标轴范围设置
        plt.ylim((0,1000))#纵坐标轴范围设置
        for k in range(0,10000):
            if abs(x_r[j,k])>3e3 or abs(y_r[j,k])>3e3:
                if k not in missing_list:
                    tan_theta[0,k]=((y_r[j+1,k]-y_r[j,k])/(x_r[j+1,k]-x_r[j,k]))
                    if math.atan(tan_theta[0,k])<0  and  x_r[j,k]<0:#第二象限
                        theta_hist.append((math.atan(tan_theta[0,k])+np.pi)/np.pi*180)
                    elif math.atan(tan_theta[0,k])<0  and  x_r[j,k]>0:#第四象限
                        theta_hist.append(abs(math.atan(tan_theta[0,k]))/np.pi*180)
                    elif math.atan(tan_theta[0,k])>0  and  x_r[j,k]>0:#第一象限
                        theta_hist.append((math.atan(tan_theta[0,k]))/np.pi*180)
                    elif math.atan(tan_theta[0,k])>0  and  x_r[j,k]<0:#第三象限
                        theta_hist.append((np.pi-math.atan(tan_theta[0,k]))/np.pi*180)
                    missing_list.append(k) 
        plt.hist(theta_hist,bins=180)
        n,bins,pach=plt.hist(theta_hist,bins=180)
        k=n[0]*np.sin(bins[1]*np.pi/180)**2
        plt.plot(bins[1::],k*(np.sin(bins[1::]*np.pi/360))**-2)
        display.clear_output(wait=True)
        plt.pause(0.1)
    plt.show

#小程序界面布局
L1=Label(root,text='入射能量(MeV)')
L2=Label(root,text='瞄准距离(fm)')
L3=Label(root,text='帧数(frames/s)')
L1.grid(row=0,column=1)
L2.grid(row=0,column=2)
L3.grid(row=0,column=3)
B1=Button(root,text='手动输入绘图',width=20,fg='ivory',bg='saddlebrown')#创建手动输入绘图按钮
B1.bind('<Button-1>',draw)#绑定手动输入绘图函数
B2=Button(root,text='随机生成绘图',width=20,fg='ivory',bg='saddlebrown')#创建随机生成绘图按钮
B2.bind('<Button-1>',random_draw)#绑定随机生成绘图函数
B1.grid(row=7,column=1)
B2.grid(row=7,column=2)
S1.grid(row=1,column=3,rowspan=6)
#利用循环把输入框布局
for i in range(0,6):
    particlerank['R'+str(i)].grid(row=i+1,column=0)
    energy['E'+str(i)].grid(row=i+1,column=1)
    aimdistance['A'+str(i)].grid(row=i+1,column=2)   
root.mainloop()#显示窗口