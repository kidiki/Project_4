'''
This program simulates a metal plate heated from the top and bottom and kept with fixed temperature on the edges. 
It uses the finite difference approximation to estimate the derivatives of the heat equation numerically.
It generates two videos to show how the time steps is crucial for a meaningful solution. 
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
start=time.time() #timing the code
print('This code takes around 3 min to run, please be patient')
#  constants
L = 0.01 #length of the plate in meters
D = 4.25e-6 # the heat diffusion constant, units: m2 s-1
N = 100 #number of points on the plate in every row and column 
dx = L/N #distance between the points on the grid

fig=plt.figure() 

#Creates the function which takes the time steps as an argument to calculate the evolution of temp on the plate 
def heat(dt):
    #  create arrays and sets the initial temperatures
    Tlo, Tmid, Thi = 200.0, 250.0, 400.0  #  initial temperatures in K
    T = Tmid*np.ones((N+1,N+1),float) #Sets the middle temperature
    #sets the boundary conditions
    T[0,:] = Thi
    T[N,:] = Thi
    T[:,0] = Tlo
    T[:,N] = Tlo
    Tp = Tmid*np.ones((N+1,N+1),float) #Sets the middle temperature
    #sets the boundary conditions
    Tp[0,:] = Thi
    Tp[N,:] = Thi
    Tp[:,0] = Tlo
    Tp[:,N] = Tlo
    t = 0.0 
    c = dt*D/(dx**2) 
    ims=[]
    k=0
    while t<10: 
        Tp[1:N,1:N]=T[1:N,1:N]+c*(T[1:N,2:N+1]+T[1:N,0:N-1]-4*T[1:N,1:N]+T[2:N+1,1:N]+T[0:N-1,1:N]) # calculates the new values of T at time t
        T[1:N,1:N],Tp[1:N,1:N] = Tp[1:N,1:N],T[1:N,1:N] #Interchanges the initial and changed array of temperatures
        t += dt
        k=k+1
        if (k%100==0):
            ims.append((plt.imshow(np.copy(T)),)) #save the state of the plate every 100 steps
    return ims

#Creates and saves the two animations
imani_1 = animation.ArtistAnimation(fig,heat(1e-4),interval=30,repeat=False) 
imani_1.save('heat2d_converged.mp4')
plt.clf()
imani_2 = animation.ArtistAnimation(fig,heat(6*1e-4),interval=30,repeat=False) 
imani_2.save('heat2d_diverged.mp4')

end=time.time()
print("The programme was running for: ", end-start)