#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spl

# In[2]:


import cupyx.scipy.sparse as css
import cupyx.scipy.sparse.linalg as cssl

# In[3]:


def laplacian1D(N, dx):
    diag=np.ones(N)
    mat=sp.spdiags(np.vstack([-diag,16*diag,-30*diag,16*diag,-diag]),[-2,-1,0,1,2],N,N)
    return mat/(12*dx**2)

def laplacian2D(N, dx):
    diag=np.ones([N*N])
    mat=sp.spdiags(np.vstack([-diag,16*diag,-30*diag,16*diag,-diag]),[-2,-1,0,1,2],N,N,'dia')
    I=sp.eye(N,format='dia')
    return (sp.kron(I,mat,format='dia')+sp.kron(mat,I,format='dia'))/(12*dx**2)

# In[4]:


def GPUlaplacian1D(N, dx):
    diag=cp.ones(N)
    mat=css.spdiags([-diag,16*diag,-30*diag,16*diag,-diag],[-2,-1,0,1,2],N,N,'csc')/(12*dx**2)
    return mat

def GPUlaplacian2Dold(N, dx):
    diag=cp.ones([N])
    mat=css.spdiags([-diag,16*diag,-30*diag,16*diag,-diag],[-2,-1,0,1,2],N,N,'dia')/(12*dx**2)
    I=css.eye(N,format='dia')
    return css.kron(I,mat)+css.kron(mat,I)

def GPUlaplacian2D(N, dx):
    diag = cp.ones([N*N])/(12*dx**2)
    diag1 = cp.copy(diag)
    diag1[(N-1)::N] = 0
    diag1m = cp.roll(diag1, 0)
    diag1p = cp.roll(diag1, 1)

    diag2 = cp.copy(diag)
    diag2[(N-2)::N] = 0
    diag2[(N-1)::N] = 0
    diag2m = cp.roll(diag2, 0)
    diag2p = cp.roll(diag2, 2)

    mat=css.spdiags([-diag,16*diag,-diag2m,16*diag1m,-60*diag,16*diag1p,-diag2p,16*diag,-diag],
                    [-2*(N-1)-2,-(N-1)-1,-2,-1,0,1,2,(N-1)+1,2*(N-1)+2],N**2,N**2,'dia')
    return mat

def GPUkinterm(N, dx, dt):
    diag = (-1j)*(dt/2)*(-0.5)*cp.ones([N*N])/(12*dx**2)
    diag1 = cp.copy(diag)
    diag1[(N-1)::N] = 0
    diag1m = cp.roll(diag1, 0)
    diag1p = cp.roll(diag1, 1)

    diag2 = cp.copy(diag)
    diag2[(N-2)::N] = 0
    diag2[(N-1)::N] = 0
    diag2m = cp.roll(diag2, 0)
    diag2p = cp.roll(diag2, 2)

    mat=css.spdiags([-diag,16*diag,-diag2m,16*diag1m,-60*diag,16*diag1p,-diag2p,16*diag,-diag],
                    [-2*(N-1)-2,-(N-1)-1,-2,-1,0,1,2,(N-1)+1,2*(N-1)+2],N**2,N**2,'dia')
    return mat

# In[5]:


# number of grid points
n = 1201

# spatial extent of grid: [[La,Lb]
La = -80
Lb = 40

# 1d grid
xgrid = cp.linspace(La, Lb, n)
xgridCPU = np.linspace(La, Lb, n)
dx = (Lb-La)/(n-1)
print(dx)

# 2d grid
xmat, ymat = cp.meshgrid(xgrid, xgrid)

# In[6]:


# testfun = xgrid**3
# print(testfun.shape)
# gpulap = GPUlaplacian1D(n, dx)
# print(type(gpulap))
# testlap = gpulap.dot( testfun )
# print(testlap.shape)
# print( cp.linalg.norm(testlap[2:-2] - 6*xgrid[2:-2]) )

# In[7]:


vmat = -((xmat + 10)**2 + 1)**(-0.5) - ((ymat + 10)**2 + 1)**(-0.5) + ((xmat-ymat)**2 + 1)**(-0.5)
print(type(vmat))
print(vmat.shape)

# In[8]:


# time step
# natural units of time
tunits = 2.4188843265857e-17
# want dt*tunits to be 2.4e-3 femtoseconds
dt = 0.01*2.4e-3*1e-15/tunits

print(dt)
print(dt/dx**2)

# number of steps
nsteps = 60000

# monitor normalization ever normint frames
normint = 200

# save every saveint frames
saveint = 1

# In[9]:


# set up one electron in the soft Coulomb external potential
ham1d = -0.5*laplacian1D(n, dx) - sp.spdiags( np.array( [((xgrid.get() + 10)**2 + 1)**(-0.5)] ), 0, n, n )

# compute ground state
eval, evec = spl.eigsh(ham1d, k=1, which='SA')

# normalization constant is 1/sqrt(dx)
# numgs = numerical ground state
numgs = dx**(-0.5)*evec[:,0]
print(np.sum(np.square(np.abs(numgs)))*dx)

# check that it's an eigenfunction
print(np.linalg.norm( ham1d@numgs - eval[0]*numgs ))

# In[10]:


# define Gaussian wavepacket
def phiWP(x,alpha,x0,p):
    phi = ((2*alpha/np.pi)**(0.25))*np.exp(-alpha*(x-x0)**2 + (1j)*p*(x-x0)) 
    return phi

# initial condition
# psiold = cp.zeros(n**2, dtype=cp.complex128)
phiWPvec = phiWP(xgrid.get(),0.1,10.0,-1.5)
psiold = cp.array( ( 1.0/np.sqrt(2.0)*(np.outer(phiWPvec, numgs) + np.outer(numgs, phiWPvec)) ).reshape((-1)) )

# make sure things are normalized
print(np.sum(np.square(np.abs(psiold))*dx*dx))
normalizer = ( np.sum(np.square(np.abs(psiold))*dx*dx ) )**(-1/2)
psiold *= normalizer
print(np.sum(np.square(np.abs(psiold))*dx*dx))

# set up array needed for iteration
psinew = cp.zeros(n**2, dtype=cp.complex128)

# In[11]:


gpukinterm = GPUkinterm(n, dx, dt)
gpukinprop = css.spdiags(cp.ones(n**2),[0],n**2,n**2,'dia') + gpukinterm + (gpukinterm @ gpukinterm)/2.0 + (gpukinterm @ gpukinterm @ gpukinterm)/6.0 + (gpukinterm @ gpukinterm @ gpukinterm @ gpukinterm)/24.0
del gpukinterm


# In[12]:


gpupotprop = css.spdiags([cp.exp((-1j)*dt*vmat.reshape((-1)))],[0],n**2,n**2)

# In[13]:


current = np.zeros((nsteps+1, n))
density = np.zeros((nsteps+1, n))

psioldC = psiold.get()
psioldM = psioldC.reshape((n,n))
gradPSIr = np.gradient(np.real(psioldM), dx, axis=0)
gradPSIi = np.gradient(np.imag(psioldM), dx, axis=0)
current[0,:] = np.trapz(2*(np.real(psioldM)*gradPSIi-np.imag(psioldM)*gradPSIr), xgrid.get(), axis=1)
density[0,:] = 2*np.trapz(np.abs(psioldM)**2,xgrid.get(),axis=1)


# In[14]:


for j in range(nsteps):
    
    # operator splitting time step
    psinew[:] = gpukinprop @ (gpupotprop @ (gpukinprop @ psiold))
    
    # monitor normalization
    if j % normint == 0:
        print( [j, cp.sum(cp.square(cp.abs(psinew))*dx*dx)] )
        
    # compute and store current density and electron density
    psinewC = psinew.get()
    psinewM = psinewC.reshape((n,n))
    gradPSI = np.gradient(psinewM, dx, axis=0)
    current[j+1,:] = np.trapz(2*(np.real(psinewM)*np.imag(gradPSI)-np.imag(psinewM)*np.real(gradPSI)), xgridCPU, axis=1)
    density[j+1,:] = 2*np.trapz(np.abs(psinewM)**2,xgridCPU,axis=1)

    psiold[:] = psinew

# In[15]:


# save to disk
np.savez('./datap15/testGPUcurrents.npz',current=current)
np.savez('./datap15/testGPUdensities.npz',density=density)

# In[1]:


# In[17]:


#         
#     if j % saveint == 0:
#         psinewC = psinew.get()
#         psinewM = psinewC.reshape((n,n))
#         plt.figure(figsize=(8,8))
#         plt.contourf(xmat.get(), ymat.get(), np.abs(psinewM)**2, levels=50, cmap='turbo')
#         plt.gca().set_aspect(1.0)
#         framenum = str(j//saveint)
#         fname = './fraGPU/frame' + framenum.zfill(5) + '.jpg'
#         plt.savefig(fname)
#         plt.close()
        
#         # compute and save the current density
#         gradPSIr = np.gradient(np.real(psinewM), dx, axis=0)
#         gradPSIi = np.gradient(np.imag(psinewM), dx, axis=0)
#         current = np.trapz(2*(np.real(psinewM)*gradPSIi-np.imag(psinewM)*gradPSIr), 
#                            xgrid.get(), axis=1)
#         fname = './denGPU/cur' + framenum.zfill(5) + '.npz'
#         np.savez(file=fname, current=current)
        
#         # only save the latest wave function
#         fname = './fraGPU/framelatest.npz'
#         np.savez(file=fname, psinew=psinew)
        
#         density=2*np.trapz(np.abs(psinewM)**2,xgrid.get(),axis=1)
#         plt.figure(figsize=(8,8))
#         plt.plot(xgrid.get(), density, color='black')
#         plt.ylim(0,0.5)
#         fname = './denGPU/den' + framenum.zfill(5) + '.jpg'
#         plt.savefig(fname)
#         plt.close()
        
#         # save all the densities
#         fname = './denGPU/den' + framenum.zfill(5) + '.npz'
#         np.savez(file=fname, density=density)
    

# In[ ]:



