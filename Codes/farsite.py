#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 01:48:02 2023

@author: Enrique Pendás-Recondo

Farsite, native units
"""

# Load libraries

import time
import numpy as np
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

tstart = time.time()





##### Inputs

### Dimensions (t,x,y)
ti,tf,nt = [0.0, 1, 10] # Time coordinate
xi,xf,nx = [-5, 5, 50] # x coordinate
yi,yf,ny = [-5, 5, 50] # y coordinate
delta = 0.0001 # Precision in exact derivatives

t = np.linspace(ti,tf,nt)
x = np.linspace(xi,xf,nx)
y = np.linspace(yi,yf,ny)
tt,xx,yy = np.meshgrid(t,x,y, indexing='ij')


### Initial parameters
ncurves = 50 # Number of trajectories to be computed
nfronts = 4 # Number of fronts to be shown
steps = 12 # Number of steps for the ODE solver
point = False # If 'True', the initial front is a point. If 'False', it is a closed curve
if (point):
    p0 = [-1.0,1.0] # Initial point
else:
    theta = np.linspace(0,2*np.pi,21)
    frontx,fronty = [0.2*np.cos(theta)-1,0.2*np.sin(theta)+1] # Initial curve counter-clockwise parametrized
    out = True # If 'True', outward trajectories are computed. If 'False', inward ones are computed


### Rothermel's model
# Fuel particle properties
H = 8000*np.ones([nt,nx,ny]) # Heat content
S_t = 0.0555*np.ones([nt,nx,ny]) # Total mineral content
S_e = 0.010*np.ones([nt,nx,ny]) # Effective mineral content
Rho_p = 32*np.ones([nt,nx,ny]) # Particle density

# Fuel array properties
Sigma = 1000*np.ones([nt,nx,ny]) # Surface area to volume ratio
W_0 = 10*np.ones([nt,nx,ny]) # Fuel load
Delta = 5*np.ones([nt,nx,ny]) # Fuel bed depth
M_x = 1*np.ones([nt,nx,ny]) # Dead fuel moisture of extinction

# Environmental values
z = np.exp(-np.square(xx[0])/2-np.square(yy[0])/2) # Surface topography (height map)
dzx,dzy = np.gradient(z,x,y)
phi_s = np.arccos(np.divide(1,np.sqrt(1+np.square(dzx)+np.square(dzy)))) # Slope steepness
M_f = 0.5*np.ones([nt,nx,ny]) # Fuel moisture
U = 1*np.ones([nt,nx,ny]) # Midflame wind speed
phi_w = 0.0*np.ones([nt,nx,ny]) # Wind direction wrt the x-axis (on the tangent plane to the surface)





##### Definition of functions

### 1-form 'omega'
def omega(v1,v2,dzx,dzy,phi,alpha_v,beta_v):
    w1 = v1+np.multiply(dzx,beta_v)
    w2 = np.sqrt(1+np.square(dzx))
    w3 = np.multiply(np.divide(w1,w2),np.cos(phi))
    w4 = v2+np.multiply(dzy,beta_v)
    w5 = np.sqrt(1+np.square(dzy))
    w6 = np.multiply(np.divide(w4,w5),np.sin(phi))
    return w3+w6


### Finsler metric
def F(v1,v2,dzx,dzy,phi,epsilon,a,h):
    beta_v = v1*dzx+v2*dzy
    alpha_v =  np.sqrt(v1**2+v2**2+np.square(beta_v))
    omega_v = omega(v1,v2,dzx,dzy,phi,alpha_v,beta_v)
      
    f1 = np.square(alpha_v)
    f2 = np.multiply(f1,np.multiply(a,(1-np.square(epsilon))))
    f3 = alpha_v-np.multiply(epsilon,omega_v)
    f4 = np.multiply(h,alpha_v+beta_v)
    return np.divide(f1,np.divide(f2,f3)+f4)


### Vector function that defines the ODE system
def f(x0,x1,x2,v1,v2,t,x,y,dzx,dzy,phi,epsilon,a,h,delta):
    nt,nx,ny = [len(t),len(x),len(y)]
    zero = np.zeros([nt,nx,ny])
    one = np.ones([nt,nx,ny])

    # Fundamental tensor g
    g11 = (2*np.square(F(v1,v2,dzx,dzy,phi,epsilon,a,h))
           -np.square(F(v1+2*delta,v2,dzx,dzy,phi,epsilon,a,h))
           -np.square(F(v1-2*delta,v2,dzx,dzy,phi,epsilon,a,h)))/(8*delta**2)
    g12 = (np.square(F(v1+delta,v2-delta,dzx,dzy,phi,epsilon,a,h))
           +np.square(F(v1-delta,v2+delta,dzx,dzy,phi,epsilon,a,h))
           -np.square(F(v1+delta,v2+delta,dzx,dzy,phi,epsilon,a,h))
           -np.square(F(v1-delta,v2-delta,dzx,dzy,phi,epsilon,a,h)))/(8*delta**2)
    g22 = (2*np.square(F(v1,v2,dzx,dzy,phi,epsilon,a,h))
           -np.square(F(v1,v2+2*delta,dzx,dzy,phi,epsilon,a,h))
           -np.square(F(v1,v2-2*delta,dzx,dzy,phi,epsilon,a,h)))/(8*delta**2)
    
    # Inverse matrix of g
    det=np.multiply(g11,g22)-np.square(g12)
    g00inv = one
    g01inv = zero
    g02inv = zero
    g11inv = np.divide(g22,det)
    g12inv = np.divide(-g12,det)
    g22inv = np.divide(g11,det)
    ginv =  np.array([[g00inv,g01inv,g02inv],[g01inv,g11inv,g12inv],[g02inv,g12inv,g22inv]])
    
    # g-derivatives
    dg11 = np.gradient(g11,t,x,y)
    dg12 = np.gradient(g12,t,x,y)
    dg22 = np.gradient(g22,t,x,y)
    dgt = np.array([[zero,zero,zero],[zero,dg11[0],dg12[0]],
                    [zero,dg12[0],dg22[0]]])
    dgx = np.array([[zero,zero,zero],[zero,dg11[1],dg12[1]],
                    [zero,dg12[1],dg22[1]]])
    dgy = np.array([[zero,zero,zero],[zero,dg11[2],dg12[2]],
                    [zero,dg12[2],dg22[2]]])
    dg = np.array([dgt,dgx,dgy])
    
    # Formal Christoffel symbols
    gamma = np.zeros([3,3,3,nt,nx,ny])
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for r in range(3):
                    s = 1/2*np.multiply(ginv[k,r],dg[i,r,j]+dg[j,r,i]-dg[r,i,j])
                    gamma[k,i,j] = gamma[k,i,j]+s
    
    # Vector function to be used in the ODE solver
    Y = np.array([1.0,v1,v2])
    f1 = v1
    f2 = v2
    f3 = 0
    f4 = 0
    for i in range(3):
        for j in range(3):
            gamma0 = interpolate.interpn([t,x,y],gamma[0,i,j],[x0,x1,x2],method='linear')[0]
            gamma1 = interpolate.interpn([t,x,y],gamma[1,i,j],[x0,x1,x2],method='linear')[0]
            gamma2 = interpolate.interpn([t,x,y],gamma[2,i,j],[x0,x1,x2],method='linear')[0]
            f3 = f3-gamma1*Y[i]*Y[j]+gamma0*Y[i]*Y[j]*Y[1]
            f4 = f4-gamma2*Y[i]*Y[j]+gamma0*Y[i]*Y[j]*Y[2]
    return np.array([f1,f2,f3,f4])


### Orthogonality equation
def eq(ang,dsf1,dsf2,dzx,dzy,phi,epsilon,a,h,t,x,y,t0,x0,y0):
    dtf1 = np.cos(ang)
    dtf2 = np.sin(ang) # Unitary wrt the Euclidean metric
    beta_t = dtf1*dzx+dtf2*dzy
    beta_s = dsf1*dzx+dsf2*dzy
    alpha_t = np.sqrt(1+np.square(beta_t))
    alpha_s = np.sqrt(dsf1**2+dsf2**2+np.square(beta_s))
        
    omega_t = omega(dtf1,dtf2,dzx,dzy,phi,alpha_t,beta_t)
    omega_s = omega(dsf1,dsf2,dzx,dzy,phi,alpha_s,beta_s)
    dot = dtf1*dsf1+dtf2*dsf2+np.multiply(beta_t,beta_s)
    
    p1 = np.multiply(a,1-np.square(epsilon))
    p2 = alpha_t-np.multiply(epsilon,omega_t)
    p3 = 2*np.multiply(dot,omega_t)-np.multiply(np.square(alpha_t),omega_s)
    p4 = np.multiply(epsilon,p3)-np.multiply(alpha_t,dot)
    p5 = 2*np.divide(dot,np.square(alpha_t))
    p6 = np.multiply(np.square(alpha_t),np.divide(p1,p2))+np.multiply(h,alpha_t+beta_t)
    p7 = np.multiply(h,np.divide(dot,alpha_t)+beta_s)
    
    eq1_mat = np.multiply(np.divide(p1,np.square(p2)),
              p4)+np.multiply(p5,p6)-p7 # F-orthogonal to the initial front
    eq1 = interpolate.interpn([t,x,y],eq1_mat,[t0,x0,y0],method='linear')[0]
    return eq1


### Jacobian of the orthogonality equation
def jaceq(ang,dsf1,dsf2,dzx,dzy,phi,epsilon,a,h,t,x,y,t0,x0,y0):
    dtf1 = np.cos(ang)
    dtf2 = np.sin(ang)
    beta_t = dtf1*dzx+dtf2*dzy
    beta_s = dsf1*dzx+dsf2*dzy
    alpha_t =  np.sqrt(1+np.square(beta_t))
    alpha_s = np.sqrt(dsf1**2+dsf2**2+np.square(beta_s))
        
    omega_t = omega(dtf1,dtf2,dzx,dzy,phi,alpha_t,beta_t)
    omega_s = omega(dsf1,dsf2,dzx,dzy,phi,alpha_s,beta_s)
    dot = dtf1*dsf1+dtf2*dsf2+np.multiply(beta_t,beta_s)
    
    p1 = np.multiply(dzx,beta_t)+dtf1
    q1 = np.multiply(dzy,beta_t)+dtf2
    p2 = np.multiply(a,1-np.square(epsilon))
    p3 = alpha_t-np.multiply(epsilon,omega_t)
    p4 = np.divide(p1,alpha_t)
    q4 = np.divide(q1,alpha_t)
    p5 = np.divide(np.multiply(np.multiply(dzx,dzy),np.sin(phi)),
                   np.sqrt(np.square(dzy)+1))+np.multiply(np.sqrt(np.square(dzx)+1),np.cos(phi))
    q5 = np.divide(np.multiply(np.multiply(dzy,dzx),np.cos(phi)),
                   np.sqrt(np.square(dzx)+1))+np.multiply(np.sqrt(np.square(dzy)+1),np.sin(phi))
    
    r1 = np.multiply(dzx,beta_s)+dsf1
    s1 = np.multiply(dzy,beta_s)+dsf2
    r2 = np.multiply(p2,np.square(alpha_t))
    r3 = np.divide(r2,p3)
    r4 = np.multiply(h,alpha_t+beta_t)
    rr1 = 2*np.multiply(r1,r3+r4)
    ss1 = 2*np.multiply(s1,r3+r4)
    
    r5 = dot*(r3+r4)
    rr2 = 4*np.multiply(p1,r5)
    ss2 = 4*np.multiply(q1,r5)
    
    r6 = 2*np.divide(np.multiply(p2,p1),p3)
    s6 = 2*np.divide(np.multiply(p2,q1),p3)
    r7 = np.multiply(r2,p4-np.multiply(epsilon,p5))
    s7 = np.multiply(r2,q4-np.multiply(epsilon,q5))
    rr3 = 2*dot*(r6-np.divide(r7,np.square(p3))+np.multiply(h,p4+dzx))
    ss3 = 2*dot*(s6-np.divide(s7,np.square(p3))+np.multiply(h,q4+dzy))
    
    r8 = 2*np.multiply(p2,p4-np.multiply(epsilon,p5))
    s8 = 2*np.multiply(p2,q4-np.multiply(epsilon,q5))
    r9 = np.multiply(epsilon,2*dot*omega_t-np.multiply(omega_s,np.square(alpha_t)))
    rr4 = np.multiply(r8,r9-dot*alpha_t)
    ss4 = np.multiply(s8,r9-dot*alpha_t)
    
    r10 = np.multiply(r1,alpha_t)
    s10 = np.multiply(s1,alpha_t)
    r11 = np.multiply(r1,omega_t)
    s11 = np.multiply(s1,omega_t)
    r12 = np.multiply(omega_s,p1)
    s12 = np.multiply(omega_s,q1)
    rr5 = np.multiply(p2,-r10-dot*p4+2*np.multiply(epsilon,r11+dot*p5-r12))
    ss5 = np.multiply(p2,-s10-dot*q4+2*np.multiply(epsilon,s11+dot*q5-s12))
    
    rr6 = np.multiply(h,p4-dot*np.divide(p1,np.power(alpha_t,3)))
    ss6 = np.multiply(h,q4-dot*np.divide(q1,np.power(alpha_t,3)))
    
    jac21_mat = np.divide(rr1,np.square(alpha_t))-np.divide(rr2,
                np.power(alpha_t,4))+np.divide(rr3,np.square(alpha_t))-np.divide(rr4,
                np.power(p3,3))+np.divide(rr5,np.square(p3))-rr6
                                                                                 
    jac22_mat = np.divide(ss1,np.square(alpha_t))-np.divide(ss2,
                np.power(alpha_t,4))+np.divide(ss3,np.square(alpha_t))-np.divide(ss4,
                np.power(p3,3))+np.divide(ss5,np.square(p3))-ss6                                                                               
    
    partd1 = -np.sin(ang)
    partd2 = np.cos(ang)
    
    totald_mat = jac21_mat*partd1+jac22_mat*partd2
    totald = interpolate.interpn([t,x,y],totald_mat,[t0,x0,y0],method='linear')[0]
    return totald





##### Geometric model

### Maximum spread rate of surface head fire (R in Rothermel's model)
Gamma_max = np.multiply(np.power(Sigma,1.5),np.power(495+0.0594*np.power(Sigma,1.5),-1)) # Maximum reaction velocity
Rho_b = np.divide(W_0,Delta) # Bulk density
Beta_op = 3.348*np.power(Sigma,-0.8189) # Optimum packing ratio
Beta = np.divide(Rho_b,Rho_p) # Packing ratio
W_n = np.multiply(W_0,1-S_t) # Net fuel load
A = 133*np.power(Sigma,-0.7913)
Gamma = np.multiply(np.multiply(Gamma_max,np.power(np.divide(Beta,Beta_op),A)),
                    np.exp(np.multiply(A,1-np.divide(Beta,Beta_op)))) # Reaction velocity
R_m = np.divide(M_f,M_x)
for i in range(nt):
    for j in range(nx):
        for k in range(ny):
            if R_m[i,j,k] > 1:
                R_m[i,j,k] = 1
Eta_m = 1-2.59*R_m+5.11*np.square(R_m)-3.52*np.power(R_m,3) # Moisture damping coefficient
Eta_s = 0.174*np.power(S_e,-0.19) # Mineral damping coefficient (max = 1.0)
for i in range(nt):
    for j in range(nx):
        for k in range(ny):
            if Eta_s[i,j,k] > 1:
                Eta_s[i,j,k] = 1
I_r = Gamma*W_n*H*Eta_m*Eta_s # Reaction intensity
Xi = np.multiply(np.power(192+0.2595*Sigma,-1),
                 np.exp(np.multiply(0.792+0.681*np.sqrt(Sigma),Beta+0.1))) # Propagation flux ratio
C = 7.47*np.exp(-0.133*np.power(Sigma,0.55))
B = 0.02526*np.power(Sigma,0.54)
E = 0.715*np.exp(-3.59e-4*Sigma)
Phi_w = np.multiply(C*np.power(U,B),np.power(np.divide(Beta,Beta_op),-E)) # Wind factor
Phi_s = np.multiply(5.275*np.power(Beta,-0.3),np.square(np.tan(phi_s))) # Slope factor
Epsilon = np.exp(np.divide(-138,Sigma)) # Effective heating number
Q_ig = 250+1116*M_f # Heat of preignition

R_0 = np.divide(I_r*Xi,Rho_b*Epsilon*Q_ig) # Maximum spread rate without wind and slope
R = np.multiply(R_0,1+Phi_w+Phi_s) # Maximum spread rate


### Farsite parameters
LB = 0.936*np.exp(0.2566*U)+0.461*np.exp(-0.1548*U)-0.397 # Length to breadth ratio
for i in range(nt):
    for j in range(nx):
        for k in range(ny):
            if LB[i,j,k] > 8:
                LB[i,j,k] = 8
HB = np.divide(LB+np.sqrt(np.square(LB)-1),LB-np.sqrt(np.square(LB)-1)) # Head to back ratio
WSv = np.array([Phi_w*np.cos(phi_w)+Phi_s*np.cos(phi_s),
                Phi_w*np.sin(phi_w)+Phi_s*np.sin(phi_s)]) # Wind-slope vector


### Geometric fire spread pattern
h = np.zeros([nt,nx,ny]) # Radius of the sphere
phi = np.arctan2(WSv[1],WSv[0]) # Orientation of the ellipse (wind-slope direction)
a = (R+np.divide(R,HB))/2 # Semi-major axis of the ellipse
epsilon = np.divide(HB-1,HB+1) # Eccentricity of the ellipse





##### ODE solver: Runge-Kutta 4

### Initial conditions
T = np.linspace(ti,tf,steps+1)
deltaT = (tf-ti)/steps
sol = np.zeros([ncurves,4,steps+1])
F_vin = 0

if (point):
    # Initial conditions if the initial front is a point
    x1i,x2i = p0
    y1i = np.cos(2*np.pi/ncurves*np.linspace(0,ncurves-1,ncurves))
    y2i = np.sin(2*np.pi/ncurves*np.linspace(0,ncurves-1,ncurves))
  
    for i in range(ncurves):
        F_vin = F(y1i[i],y2i[i],dzx, dzy, phi, epsilon, a, h)
        sol[i,:,0] = [x1i,
                      x2i,
                      y1i[i]/interpolate.interpn([t,x,y],F_vin,[T[0],x1i,x2i],method='linear')[0],
                      y2i[i]/interpolate.interpn([t,x,y],F_vin,[T[0],x1i,x2i],method='linear')[0]]
else:
    # Initial conditions if the initial front is a curve
    tck,u = interpolate.splprep([frontx,fronty],k=3,s=0) # u parameter of the curve interpolating the points of the inital front
    uu = np.linspace(0,1,ncurves+1) # uu parameter of the selected points in the front (initial position of the trajectories)
    front = interpolate.splev(uu,tck)
    dfront = interpolate.splev(uu,tck,der=1)
    for i in range(ncurves):
        # Parameters for the while loop
        stop = False
        max_iter = 8
        mult = 0
        
        dsf1,dsf2 = [dfront[0][i],dfront[1][i]]
        seed = np.cross(np.array([dsf1,dsf2]),np.array([0,0,1]))
        seed_vec = np.array([seed[0],seed[1]])/np.sqrt(seed[0]**2+seed[1]**2) # Seed unitary wrt the Euclidean metric
        if seed_vec[1] >= 0:
            seed_ang = np.arccos(seed_vec[0])
        else:
            seed_ang = -np.arccos(seed_vec[0])
        
        while (not stop and max_iter > 0):
            root = optimize.root_scalar(eq,
                             args=(dsf1,dsf2,dzx,dzy,phi,epsilon,a,h,t,x,y,ti,front[0][i],front[1][i]),
                             x0=seed_ang+mult*np.pi/4,
                             fprime=jaceq)
            ang = root.root
            dtf1 = np.cos(ang)
            dtf2 = np.sin(ang)
            det = dtf1*dsf2-dtf2*dsf1
            if (out):
                if det > 0: # Condition for the initial velocity to be pointing outwards
                    stop = True
                else:
                    max_iter -= 1
                    mult += 1
            else:
                if det < 0: # Condition for the initial velocity to be pointing inwards
                    stop = True
                else:
                    max_iter -= 1
                    mult += 1
        
        F_vin = F(dtf1,dtf2,dzx, dzy, phi, epsilon, a, h)
        F_norm = interpolate.interpn([t,x,y],F_vin,[T[0],front[0][i],front[1][i]],method='linear')[0]
        sol[i,:,0] = [front[0][i],
                      front[1][i],
                      dtf1/F_norm,
                      dtf2/F_norm]   


### Iteration of the Runge-Kutta 4 method
for i in range(ncurves):
    for n in range(steps):
        k1 = f(T[n],sol[i,0,n],sol[i,1,n],sol[i,2,n],sol[i,3,n],
               t,x,y,dzx,dzy,phi,epsilon,a,h,delta)
        
        k2 = f(T[n]+deltaT/2,sol[i,0,n]+deltaT*k1[0]/2,
               sol[i,1,n]+deltaT*k1[1]/2,
               sol[i,2,n]+deltaT*k1[2]/2,sol[i,3,n]+deltaT*k1[3]/2,
               t,x,y,dzx,dzy,phi,epsilon,a,h,delta)
        
        k3 = f(T[n]+deltaT/2,sol[i,0,n]+deltaT*k2[0]/2,
               sol[i,1,n]+deltaT*k2[1]/2,sol[i,2,n]+
               deltaT*k2[2]/2,sol[i,3,n]+deltaT*k2[3]/2,
               t,x,y,dzx,dzy,phi,epsilon,a,h,delta)
        
        k4 = f(T[n]+deltaT,sol[i,0,n]+deltaT*k3[0],sol[i,1,n]+deltaT*k3[1],
               sol[i,2,n]+deltaT*k3[2],sol[i,3,n]+deltaT*k3[3],
               t,x,y,dzx,dzy,phi,epsilon,a,h,delta)
        
        sol[i,:,n+1] = sol[i,:,n]+1/6*deltaT*(k1+2*k2+2*k3+k4)
        
        

        
       
### Graphic representation

rcs = steps//nfronts # Ratio points calculated-points shown
colors = cm.rainbow(np.linspace(0, 1, nfronts))
for i in range(ncurves):
    plt.plot(sol[i,0,:],sol[i,1,:],'b-')
    for j in range(nfronts):
        plt.plot(sol[i,0,rcs*(j+1)],sol[i,1,rcs*(j+1)],'.',color=colors[j,:])

if (point):
    plt.plot(x1i,x2i,'o',color='r')
else:
    plt.plot(front[0],front[1],'r-')
    
plt.axis([xi,xf,yi,yf])
plt.gca().set_aspect('equal', adjustable='box')
plt.contourf(x,y,np.transpose(z))
plt.show()





### Time control
 
trun = time.time() - tstart