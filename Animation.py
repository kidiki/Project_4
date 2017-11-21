import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import combinations
npoint=400
nframe=500
xmin,xmax,ymin,ymax=0,1,0,1
fig, ax = plt.subplots()
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
Dt=0.00002
r=0.01
def update_point(num):
    global x,y,vx,vy
    print(num)
    indx=np.where((x < xmin) | (x > xmax))
    indy=np.where((y < ymin) | (y > ymax))
    vx[indx]=-vx[indx]
    vy[indy]=-vy[indy]
    index=np.array(list(combinations(range(400),2)))
    xx=np.asarray(list(combinations(x,2)))
    yy=np.asarray(list(combinations(y,2)))
    dd=(xx[:,0]-xx[:,1])**2+(yy[:,0]-yy[:,1])**2
    ind_coll=np.array(list(np.where(dd<=(2*r)**2)))
    #print(dd)
    #print(ind_coll)
    #print(np.size(ind_coll)) 
    for i in range (0,np.size(ind_coll[0])):
        x_1=x[index[ind_coll[0][i]][0]]
        x_2=x[index[ind_coll[0][i]][1]]
        delta_x = np.abs ( x_1 - x_2 ) #should it be 2-1 or 1-2
        y_1=y[index[ind_coll[0][i]][0]]
        y_2=y[index[ind_coll[0][i]][1]]
        delta_y = y_1 - y_2
        v_x_1=vx[index[ind_coll[0][i]][0]] #Accessing the initial x-component velocity for the first element in the ith collision
        v_x_2=vx[index[ind_coll[0][i]][1]]
        delta_v_x = v_x_1 - v_x_2
        v_y_1=vy[index[ind_coll[0][i]][0]]
        v_y_2=vy[index[ind_coll[0][i]][1]]
        delta_v_y = v_y_1 - v_y_2
        #print('the ols vel is', v_x_1)
        blah=(delta_x * delta_v_x + delta_y * delta_v_y ) / (delta_x**2 + delta_y**2)
        vx[index[ind_coll[0][i]][0]] = v_x_1 - blah * (delta_x) 
        #print('the new velocity is',vx[index[ind_coll[0][i]][0]])
        vy[index[ind_coll[0][i]][0]] = v_y_1 - blah * (delta_y) 
    dx=Dt*vx
    dy=Dt*vy
    x=x+dx
    y=y+dy
    data=np.stack((x,y),axis=-1)
    im.set_offsets(data)

x = np.random.random(npoint)
y = np.random.random(npoint)
vx=-500.*np.ones(npoint)
vy=np.zeros(npoint)
vx[np.where(x <= 0.5)]=-vx[np.where(x <= 0.5)]
s=np.array([1])
im = ax.scatter(x,y)
im.set_sizes(s)
ani = animation.FuncAnimation(fig, update_point,nframe,interval=50,repeat=False)
ani.save('collisions.mp4')
#plt.show()
