import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import combinations
from scipy.optimize import curve_fit

Dt=0.00002
r=0.01
m=2.672E-26
k=1.38064852E-23 #Boltzmann constant, units = m*2*kg*s*-2*K

npoint=400
nframe=500
xmin,xmax,ymin,ymax=0,1,0,1
fig, ax = plt.subplots()
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

x = np.random.random(npoint)
y = np.random.random(npoint)

vx=-500.*np.ones(npoint)
vy=np.zeros(npoint)
vx[np.where(x <= 0.5)]=-vx[np.where(x <= 0.5)]
s=np.array([10])
im = ax.scatter(x,y)
im.set_sizes(s)
left=np.where((x<0.5))
right=np.where((x>0.5))
c = np.zeros(npoint * 3).reshape(npoint,3)
c[left]=[0,0,1] #RGB=0,0,1 makes sll the dots on the left blue
c[right]=[1,0,0] #RGB=1,0,0 makes sll the dots on the left red
im.set_facecolor(c)

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
    ind_coll=np.array(list(np.where(dd<=( 2*0.00001)))) 
    for i in range (0,np.size(ind_coll[0])):
        x_1=x[index[ind_coll[0][i]][0]]
        x_2=x[index[ind_coll[0][i]][1]]
        delta_x =  x_1 - x_2  
        y_1=y[index[ind_coll[0][i]][0]]
        y_2=y[index[ind_coll[0][i]][1]]
        delta_y = y_1 - y_2
        v_x_1=vx[index[ind_coll[0][i]][0]] #Accessing the initial x-component velocity for the first element in the ith collision
        v_x_2=vx[index[ind_coll[0][i]][1]]
        delta_v_x = v_x_1 - v_x_2
        v_y_1=vy[index[ind_coll[0][i]][0]]
        v_y_2=vy[index[ind_coll[0][i]][1]]
        delta_v_y = v_y_1 - v_y_2
        blah=(delta_x * delta_v_x + delta_y * delta_v_y ) / (delta_x**2 + delta_y**2)
        #updating velocity of the first particle
        vx[index[ind_coll[0][i]][0]] = v_x_1 - blah * (delta_x) 
        vy[index[ind_coll[0][i]][0]] = v_y_1 - blah * (delta_y) 
        #updating velocity of the second particle
        vx[index[ind_coll[0][i]][1]] = v_x_2 + blah * (delta_x)
        vy[index[ind_coll[0][i]][1]] = v_y_2 + blah * (delta_y)
    dx=Dt*vx
    dy=Dt*vy
    x=x+dx
    y=y+dy
    data=np.stack((x,y),axis=-1)
    im.set_offsets(data)


ani = animation.FuncAnimation(fig, update_point,nframe,interval=30,repeat=False)
#ani.save('collisions.mp4')
plt.show()
plt.clf()

#Generating the plots
v=[]
v=np.sqrt(vx**2+vy**2) 
KE=(m/2)*v**2
def f(v,T):
    return ( (m*v) / (k*T) ) * np.exp( (-0.5*m*v**2) / (k*T) )   
plt.subplot(211)
#plt.hist(v, bins=20, histtype='step',normed=True)
y, bins, patches = plt.hist(v,bins=40,normed=True)
plt.xlabel('v [m/s]')
plt.ylabel('Probability')

bin_centers = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
print(bin_centers,y)
popt, pcov = curve_fit(f, bin_centers, y)
T_anal=popt
print()
plt.plot(v,f(v,T_anal),'o')

plt.subplot(212)
plt.hist(KE, bins=20, histtype='step',normed=True)
plt.xlabel('KE')
plt.ylabel('Probability')


plt.show()
#plt.savefig('distributions.pdf')



#plt.clf()
#plt.scatter(bin_centers, f(bin_centers, *popt))
#plt.show()


