from jax.config import config
config.update("jax_enable_x64", True)

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax.numpy as jnp
from jax import jit, vmap, lax

from functools import partial

import scipy.linalg as sl
import scipy.integrate as si
import scipy.optimize as so
import time

import numpy as np

import linearbases 


class learnschro:
    def __init__(self, biga, dt, nmax, nsteps=None,
            obj=None, ic=None, wfdata=None, a2data=None, kinmat=None):
        # spatial domain size
        self.biga = biga
        # time step (of the data)
        self.dt = dt
        # nmax = maximum Fourier mode used to solve forward problem
        # (size of Fourier basis used to solve forward problem = 2*self.nmax+1)
        self.nmax = nmax
        # no matter what, we need an initial wave function
        # if true initial wave function is unknown, we need current best estimate
        self.jainit = jnp.array(ic)
        # no matter what, we need a kinetic energy matrix (in Fourier basis)
        self.jkmat = jnp.array(kinmat)

        # parallelize the built-in correlate function over first axes
        self.vcorr = vmap(jnp.correlate, in_axes=(0,0,None))

        # there are only three choices for obj at the moment
        # if you choose the Wave Function (WF) obj, 
        # then you need training data in wfdata
        if obj=='WF': # wave function objective
            self.jamat = jnp.array(wfdata)
            self.nsteps = self.jamat.shape[0] - 1
            self.jobjtraj = jit(self.wfobjtraj)
            self.jadjgrad = jit(self.wfadjgrad)
            self.jadjhess = jit(self.wfadjhess)
        # if you choose the Amplitude Squared (A2) obj,
        # then you can supply either wave function training data in wfdata,
        # or you can supply amplitude squared training data in a2data
        if obj=='A2': # amplitude^2 objective
            if a2data is not None:
                self.jbetamat = jnp.array(a2data)
                self.jamat = None
                self.nsteps = self.jbetamat.shape[0] - 1
            else:
                self.jamat = jnp.array(wfdata)
                self.nsteps = self.jamat.shape[0] - 1
                self.jbetamat = self.vcorr(self.jamat, self.jamat, 'same') / jnp.sqrt(2 * self.biga)

            self.jobjtraj = jit(self.a2objtraj)
            self.jadjgrad = jit(self.a2adjgrad)
            self.jadjhess = jit(self.a2adjhess)
        # if you choose no objective function, 
        # then you can only solve forward problems
        # in this case, you need to pass in nsteps
        if obj is None:
            self.nsteps = nsteps
            # for convenience, define a zero array for jamat
            self.jamat = jnp.zeros((nsteps+1, 2*nmax+1))
            # this will enable you to call jobjtraj and compute WF trajectories,
            # from which A2 trajectories can easily be computed
            self.jobjtraj = jit(self.wfobjtraj)


        self.mkMPsv1 = vmap(self.mk_M_and_P,in_axes=(0,),out_axes=(0,0,))
        self.mkMPsv2 = vmap(self.mk_M_and_P,in_axes=(1,),out_axes=(2,2,))

    def wfadjgrad(self, vhatmat, ctrmats):
        # Hamiltonian matrix 
        hhatmat = self.jkmat + vhatmat
        
        # eigendecomposition and compute propagator
        hatspec, hatstates = jnp.linalg.eigh(hhatmat)
        hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        
        #solve *forward* problem
        ahatmat = jnp.concatenate([jnp.expand_dims(self.jainit,0), jnp.zeros((self.nsteps, ctrmats.shape[1]))])
        def forstep(j, loopamat):
            return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

        ahatmat = lax.fori_loop(0, self.nsteps, forstep, ahatmat)

        # compute objective function
        resid = ahatmat - self.jamat
        obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))

        # solve *adjoint* problem
        lambmat = jnp.concatenate([jnp.zeros((self.nsteps, ctrmats.shape[1])), jnp.expand_dims(ahatmat[self.nsteps,:] - self.jamat[self.nsteps,:],0)])
        def adjstep(j, looplamb):
            t = self.nsteps - j - 1
            return looplamb.at[t].set( ahatmat[t,:] - self.jamat[t,:] + hatpropH @ looplamb[t+1,:] )

        lambmat = lax.fori_loop(0, self.nsteps, adjstep, lambmat)

        # Compute the gradients
        m = ctrmats.shape[1]
        offdiagmask = jnp.ones((m, m)) - jnp.eye(m)
        expspec = jnp.exp(-1j*self.dt*hatspec)
        e1, e2 = jnp.meshgrid(expspec, expspec)
        s1, s2 = jnp.meshgrid(hatspec, hatspec)
        denom = offdiagmask * (-1j*self.dt)*(s1 - s2) + jnp.eye(m)
        mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
        prederivamat = jnp.einsum('ij,jkm,kl->ilm',hatstates.conj().T,ctrmats,hatstates) 
        derivamat = prederivamat * jnp.expand_dims(mask,2)
        alldmat = -1j*self.dt*jnp.einsum('ij,jkm,kl->mil',hatstates,derivamat,hatstates.conj().T)
        gradients = jnp.real(jnp.einsum('ij,ajk,ik->a', jnp.conj(lambmat[1:,:]), alldmat, ahatmat[:-1,:]))

        return obj, gradients, jnp.conj(lambmat[0,:])

    def wfadjhess(self, vhatmat, ctrmats):
        # Hamiltonian matrix 
        hhatmat = self.jkmat + vhatmat
        
        # eigendecomposition and compute propagator
        hatspec, hatstates = jnp.linalg.eigh(hhatmat)
        hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        
        #solve *forward* problem
        ahatmat = jnp.concatenate([jnp.expand_dims(self.jainit,0), jnp.zeros((self.nsteps, ctrmats.shape[1]))])
        def forstep(j, loopamat):
            return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

        ahatmat = lax.fori_loop(0, self.nsteps, forstep, ahatmat)

        # compute objective function
        resid = ahatmat - self.jamat
        obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))
        
        # solve *adjoint* problem
        lambmat = jnp.concatenate([jnp.zeros((self.nsteps, ctrmats.shape[1])), jnp.expand_dims(ahatmat[self.nsteps,:] - self.jamat[self.nsteps,:],0)])
        def adjstep(j, looplamb):
            t = self.nsteps - j - 1
            return looplamb.at[t].set( ahatmat[t,:] - self.jamat[t,:] + hatpropH @ looplamb[t+1] )

        lambmat = lax.fori_loop(0, self.nsteps, adjstep, lambmat)

        # Compute the gradients
        m = ctrmats.shape[1]
        offdiagmask = jnp.ones((m, m)) - jnp.eye(m)
        expspec = jnp.exp(-1j*self.dt*hatspec)
        e1, e2 = jnp.meshgrid(expspec, expspec)
        s1, s2 = jnp.meshgrid(hatspec, hatspec)
        denom = offdiagmask * (-1j*self.dt)*(s1 - s2) + jnp.eye(m)
        mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
        prederivamat = jnp.einsum('ij,jkm,kl->ilm',hatstates.conj().T,ctrmats,hatstates) 
        derivamat = prederivamat * jnp.expand_dims(mask,2)
        alldmat = -1j*self.dt*jnp.einsum('ij,jkm,kl->mil',hatstates,derivamat,hatstates.conj().T)
        gradients = jnp.real(jnp.einsum('ij,ajk,ik->a', jnp.conj(lambmat[1:,:]), alldmat, ahatmat[:-1,:]))
        
        # propagators
        # hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        # hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        
        # propagate \nabla_{\theta} a
        gradamat = jnp.zeros((self.nsteps+1, ctrmats.shape[1], ctrmats.shape[2]), dtype=np.complex128)
        def gradastep(j, loopgrada):
            return loopgrada.at[j+1].set( hatprop @ loopgrada[j,:,:] + (alldmat @ ahatmat[j,:]).T )

        gradamat = lax.fori_loop(0, self.nsteps, gradastep, gradamat)

        # propagate \nabla_{\theta} \lambda
        alldmatH = np.transpose(alldmat.conj(),axes=(2,1,0))
        gradlamb = jnp.concatenate([jnp.zeros((self.nsteps, ctrmats.shape[1], ctrmats.shape[2])), jnp.expand_dims(gradamat[self.nsteps],0)])
        def gradlambstep(j, loopgradlamb):
            t = self.nsteps - j - 1
            term1 = hatpropH @ loopgradlamb[t+1]
            term2 = jnp.einsum('ijk,j->ik', alldmatH, lambmat[t+1,:])
            return loopgradlamb.at[t].set( gradamat[t,:,:] + term1 + term2 )

        gradlamb = lax.fori_loop(0, self.nsteps, gradlambstep, gradlamb)

        hesspt1 = jnp.real(jnp.einsum('ijl,ajk,ik->al', jnp.conj(gradlamb[1:,:,:]), alldmat, ahatmat[:-1,:]))
        hesspt2 = jnp.real(jnp.einsum('ij,ajk,ikl->al', jnp.conj(lambmat[1:,:]), alldmat, gradamat[:-1,:,:]))
        res = self.purejaxhess(hatspec, -1j*self.dt*jnp.transpose(prederivamat,[2,0,1]))
        hesspt3 = jnp.real(jnp.einsum('ci,ij,abjk,lk,cl->ab',jnp.conj(lambmat[1:,:]),hatstates,res,jnp.conj(hatstates),ahatmat[:-1,:],optimize=True))
        hess = hesspt1 + hesspt2 + hesspt3
        
        return obj, gradients, jnp.conj(lambmat[0]), hess, jnp.conj(gradlamb[0])

    def purejaxhess(self, dvec, alldmat):
        dvec = lax.stop_gradient(dvec)
        alldmat = lax.stop_gradient(alldmat)
        n = alldmat.shape[1]
        jd = jnp.array(-1j*self.dt*dvec)
        jed = jnp.exp(jd)
        jedi = jnp.expand_dims(jed,1)
        jedj = jnp.expand_dims(jed,0)
        jdi = jnp.expand_dims(jd,1)
        jdj = jnp.expand_dims(jd,0)
        mask = jnp.expand_dims(jnp.eye(n), [0,1])
        out = jnp.expand_dims(alldmat,1)*jnp.expand_dims(alldmat,0)*mask*jedi
        jtmp = -jedi + jdi*jedi - jdj*jedi + jedj
        jtmp *= ((jdi!=jdj)*(jdi-jdj)**(-2))
        jtmp2 = jnp.expand_dims(alldmat,1).transpose([0,1,3,2]) * jnp.expand_dims(alldmat,0) 
        jtmp2 += jnp.expand_dims(alldmat,1) * jnp.expand_dims(alldmat,0).transpose([0,1,3,2])
        jtmp2 *= jtmp
        out += jnp.expand_dims(jnp.sum(jtmp2,axis=3),2) * mask
        
        unmask = 1-jnp.eye(n)
        # bmat[thread_number,i,i]
        # column vector with diagonal * amat .... bmat[thread_number,i,i]*amat[thread_number,i,k]
        # bmat * row vector with diagonal ....... bmat[thread_number,i,k]*amat[thread_number,k,k]
        bmatii = jnp.expand_dims(jnp.diagonal(alldmat,axis1=1,axis2=2),[0,3]) # * mask
        bmatkk = jnp.expand_dims(jnp.diagonal(alldmat,axis1=1,axis2=2),[0,2]) # * mask
        amatik = jnp.expand_dims(alldmat,1)
        bmatik = jnp.expand_dims(alldmat,0)
        amatkk = jnp.expand_dims(jnp.diagonal(alldmat,axis1=1,axis2=2),[1,2]) #* mask
        amatii = jnp.expand_dims(jnp.diagonal(alldmat,axis1=1,axis2=2),[1,3]) #* mask
        jdk = jnp.expand_dims(jd,0)
        jedk = jnp.expand_dims(jed,0)
        jdimk = jdi-jdk
        jdimkm1 = (jdi!=jdk)*(jdimk**(-1))
        jtmp3 = -jedi + jdimk*jedi + jedk
        jtmp3 *= (jdimkm1**2)
        jtmp4 = jtmp3 * (bmatii*amatik + amatii*bmatik)
        jtmp5 = -(-jedi + jedk + jdimk*jedk)
        jtmp5 *= (jdimkm1**2)
        jtmp6 = jtmp5 * (bmatik*amatkk + amatik*bmatkk)
        out2 = jtmp4*unmask
        out2 += jtmp6*unmask
        
        term1 = jnp.einsum('bij,ij,jk,ajk->abik',alldmat,jdimkm1,unmask,alldmat)
        term1 += jnp.einsum('aij,ij,jk,bjk->abik',alldmat,jdimkm1,unmask,alldmat)
        term1 *= (-jedi*jdimkm1)
        
        term2 = jnp.einsum('bij,j,ij,ajk,jk->abik',alldmat,jed,jdimkm1,alldmat,jdimkm1)
        term2 += jnp.einsum('aij,j,ij,bjk,jk->abik',alldmat,jed,jdimkm1,alldmat,jdimkm1)
        
        term3 = jnp.einsum('bij,ij,jk,ajk->abik',alldmat,unmask,jdimkm1,alldmat)
        term3 += jnp.einsum('aij,ij,jk,bjk->abik',alldmat,unmask,jdimkm1,alldmat)
        term3 *= (-jedk*jdimkm1)
        
        bigunmask = jnp.expand_dims(unmask,[0,1])
        out3 = -bigunmask*(term1 + term2 + term3)
        
        return out+out2+out3

    def wfobjtraj(self, vhatmat):
        # Hamiltonian matrix 
        hhatmat = self.jkmat + vhatmat
        
        # eigendecomposition and compute propagator
        hatspec, hatstates = jnp.linalg.eigh(hhatmat)
        hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        
        #solve *forward* problem
        ahatmat = jnp.concatenate([jnp.expand_dims(self.jainit,0), jnp.zeros((self.nsteps, 2*self.nmax+1))])
        def forstep(j, loopamat):
            return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

        ahatmat = lax.fori_loop(0, self.nsteps, forstep, ahatmat)
        
        # compute only the objective function
        resid = ahatmat - self.jamat
        obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))
        
        return obj, ahatmat

    def mk_M_and_P(self, avec):
        halflen = len(avec) // 2
        padavec = jnp.concatenate((jnp.zeros(halflen), jnp.array(avec), jnp.zeros(halflen)))
        rawmat = []
        for j in range(2 * halflen + 1):
            rawmat.append(padavec[2 * halflen - j:4 * halflen + 1 - j])

        Mmat = jnp.conjugate(jnp.array(rawmat))
        Pmat = jnp.flipud(jnp.array(rawmat))
        return Mmat, Pmat

    def a2adjgrad(self, vhatmat, ctrmats):
        # Hamiltonian matrix 
        hhatmat = self.jkmat + vhatmat
        
        # eigendecomposition and compute propagator
        hatspec, hatstates = jnp.linalg.eigh(hhatmat)
        hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        
        #solve *forward* problem
        ahatmat = jnp.concatenate([jnp.expand_dims(self.jainit,0), jnp.zeros((self.nsteps, ctrmats.shape[1]))])
        def forstep(j, loopamat):
            return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

        ahatmat = lax.fori_loop(0, self.nsteps, forstep, ahatmat)
        rhomat = self.vcorr(ahatmat, ahatmat, 'same') / jnp.sqrt(2 * self.biga)

        # compute objective function
        resid = rhomat - self.jbetamat
        obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))

        # compute all the Ms and Ps
        Ms, Ps = self.mkMPsv1(ahatmat)

        # solve *adjoint* problem
        resid2 = rhomat - self.jbetamat
        fincond = ( Ms[self.nsteps,:].conj().T @ resid2[self.nsteps,:] + (Ps[self.nsteps].conj().T @ resid2[self.nsteps,:]).conj() ) / jnp.sqrt(2 * self.biga)
        lambmat = jnp.concatenate([jnp.zeros((self.nsteps, ctrmats.shape[1])), jnp.expand_dims(fincond, 0)])
        def adjstep(j, looplamb):
            t = self.nsteps - j - 1
            jumpterm = ( Ms[t,:].conj().T @ resid2[t,:] + (Ps[t].conj().T @ resid2[t,:]).conj() ) / jnp.sqrt(2 * self.biga)
            return looplamb.at[t].set( jumpterm + hatpropH @ looplamb[t+1,:] )

        lambmat = lax.fori_loop(0, self.nsteps, adjstep, lambmat)

        # Compute the gradients
        m = ctrmats.shape[1]
        offdiagmask = jnp.ones((m, m)) - jnp.eye(m)
        expspec = jnp.exp(-1j*self.dt*hatspec)
        e1, e2 = jnp.meshgrid(expspec, expspec)
        s1, s2 = jnp.meshgrid(hatspec, hatspec)
        denom = offdiagmask * (-1j*self.dt)*(s1 - s2) + jnp.eye(m)
        mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
        prederivamat = jnp.einsum('ij,jkm,kl->ilm',hatstates.conj().T,ctrmats,hatstates) 
        derivamat = prederivamat * jnp.expand_dims(mask,2)
        alldmat = -1j*self.dt*jnp.einsum('ij,jkm,kl->mil',hatstates,derivamat,hatstates.conj().T)
        gradients = jnp.real(jnp.einsum('ij,ajk,ik->a', jnp.conj(lambmat[1:,:]), alldmat, ahatmat[:-1,:]))

        return obj, gradients, jnp.conj(lambmat[0])

    def a2adjhess(self, vhatmat, ctrmats):
        # Hamiltonian matrix 
        hhatmat = self.jkmat + vhatmat
        
        # eigendecomposition and compute propagator
        hatspec, hatstates = jnp.linalg.eigh(hhatmat)
        hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        
        #solve *forward* problem
        ahatmat = jnp.concatenate([jnp.expand_dims(self.jainit,0), jnp.zeros((self.nsteps, ctrmats.shape[1]))])
        def forstep(j, loopamat):
            return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

        ahatmat = lax.fori_loop(0, self.nsteps, forstep, ahatmat)
        rhomat = self.vcorr(ahatmat, ahatmat, 'same') / jnp.sqrt(2 * self.biga)

        # compute objective function
        resid = rhomat - self.jbetamat
        obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))
        
        # compute all the Ms and Ps
        Ms, Ps = self.mkMPsv1(ahatmat)

        # solve *adjoint* problem
        resid2 = rhomat - self.jbetamat
        fincond = ( Ms[self.nsteps,:].conj().T @ resid2[self.nsteps,:] + (Ps[self.nsteps].conj().T @ resid2[self.nsteps,:]).conj() ) / jnp.sqrt(2 * self.biga)
        lambmat = jnp.concatenate([jnp.zeros((self.nsteps, ctrmats.shape[1])), jnp.expand_dims(fincond, 0)])
        def adjstep(j, looplamb):
            t = self.nsteps - j - 1
            jumpterm = ( Ms[t,:].conj().T @ resid2[t,:] + (Ps[t].conj().T @ resid2[t,:]).conj() ) / jnp.sqrt(2 * self.biga)
            return looplamb.at[t].set( jumpterm + hatpropH @ looplamb[t+1,:] )

        lambmat = lax.fori_loop(0, self.nsteps, adjstep, lambmat)

        # Compute the gradients
        m = ctrmats.shape[1]
        offdiagmask = jnp.ones((m, m)) - jnp.eye(m)
        expspec = jnp.exp(-1j*self.dt*hatspec)
        e1, e2 = jnp.meshgrid(expspec, expspec)
        s1, s2 = jnp.meshgrid(hatspec, hatspec)
        denom = offdiagmask * (-1j*self.dt)*(s1 - s2) + jnp.eye(m)
        mask = offdiagmask * (e1 - e2)/denom + jnp.diag(expspec)
        prederivamat = jnp.einsum('ij,jkm,kl->ilm',hatstates.conj().T,ctrmats,hatstates) 
        derivamat = prederivamat * jnp.expand_dims(mask,2)
        alldmat = -1j*self.dt*jnp.einsum('ij,jkm,kl->mil',hatstates,derivamat,hatstates.conj().T)
        gradients = jnp.real(jnp.einsum('ij,ajk,ik->a', jnp.conj(lambmat[1:,:]), alldmat, ahatmat[:-1,:]))

        # propagate \nabla_{\theta} a
        gradamat = jnp.zeros((self.nsteps+1, ctrmats.shape[1], ctrmats.shape[2]), dtype=np.complex128)
        def gradastep(j, loopgrada):
            return loopgrada.at[j+1].set( hatprop @ loopgrada[j,:,:] + (alldmat @ ahatmat[j,:]).T )

        gradamat = lax.fori_loop(0, self.nsteps, gradastep, gradamat)

        # self.nsteps+1 x 2*numfour + 1
        rhomat = jnp.array(rhomat)
        
        # new pieces
        finGM, finGP = self.mkMPsv2(gradamat[self.nsteps,:,:])
        finMmat, finPmat = self.mk_M_and_P(ahatmat[self.nsteps])
        
        # term I in notes
        fincond = 1/jnp.sqrt(2*self.biga) * jnp.einsum('jki,j->ki',finGP.conj(),rhomat[self.nsteps]-self.jbetamat[self.nsteps])
        # term III in notes
        fincond += 1/jnp.sqrt(2*self.biga) * jnp.einsum('jki,j->ki',finGM,(rhomat[self.nsteps]-self.jbetamat[self.nsteps]).conj())
        # term II in notes
        fincond += 1/(2*self.biga) * jnp.einsum('jk,jl,ki->li',finMmat.conj(),finMmat,gradamat[self.nsteps].conj())
        fincond += 1/(2*self.biga) * jnp.einsum('jk,jl,ki->li',finMmat,finPmat.conj(),gradamat[self.nsteps])
        # term IV in notes
        fincond += 1/(2*self.biga) * jnp.einsum('jk,jl,ki->li',finPmat.conj(),finMmat,gradamat[self.nsteps])
        fincond += 1/(2*self.biga) * jnp.einsum('jk,jl,ki->li',finPmat,finPmat.conj(),gradamat[self.nsteps].conj())

        # propagate \nabla_{\theta} \lambda
        alldmatH = jnp.transpose(alldmat.conj(),axes=(2,1,0))
        gradlamb = jnp.concatenate([jnp.zeros((self.nsteps, ctrmats.shape[1], ctrmats.shape[2]), dtype=jnp.complex128), jnp.expand_dims(fincond.conj(),0)])
        def gradlambstep(j, loopgradlamb):
            t = self.nsteps - j - 1
            term1 = hatpropH @ loopgradlamb[t+1]
            term2 = jnp.einsum('ijk,j->ik', alldmatH, lambmat[t+1,:])
            # new pieces
            thisGM, thisGP = self.mkMPsv2(gradamat[t,:,:])
            thisMmat, thisPmat = self.mk_M_and_P(ahatmat[t])

            # term I in notes
            jumpterm = 1/jnp.sqrt(2*self.biga) * jnp.einsum('jki,j->ki',thisGP.conj(),rhomat[t]-self.jbetamat[t])
            # term III in notes
            jumpterm += 1/jnp.sqrt(2*self.biga) * jnp.einsum('jki,j->ki',thisGM,(rhomat[t]-self.jbetamat[t]).conj())
            # term II in notes
            jumpterm += 1/(2*self.biga) * jnp.einsum('jk,jl,ki->li',thisMmat.conj(),thisMmat,gradamat[t].conj())
            jumpterm += 1/(2*self.biga) * jnp.einsum('jk,jl,ki->li',thisMmat,thisPmat.conj(),gradamat[t])
            # term IV in notes
            jumpterm += 1/(2*self.biga) * jnp.einsum('jk,jl,ki->li',thisPmat.conj(),thisMmat,gradamat[t])
            jumpterm += 1/(2*self.biga) * jnp.einsum('jk,jl,ki->li',thisPmat,thisPmat.conj(),gradamat[t].conj())
            return loopgradlamb.at[t].set( jumpterm.conj() + term1 + term2 )

        gradlamb = lax.fori_loop(0, self.nsteps, gradlambstep, gradlamb)

        hesspt1 = jnp.real(jnp.einsum('ijl,ajk,ik->al', jnp.conj(gradlamb[1:,:,:]), alldmat, ahatmat[:-1,:]))
        hesspt2 = jnp.real(jnp.einsum('ij,ajk,ikl->al', jnp.conj(lambmat[1:,:]), alldmat, gradamat[:-1,:,:]))
        res = self.purejaxhess(hatspec, -1j*self.dt*jnp.transpose(prederivamat,[2,0,1]))
        hesspt3 = jnp.real(jnp.einsum('ci,ij,abjk,lk,cl->ab',jnp.conj(lambmat[1:,:]),hatstates,res,jnp.conj(hatstates),ahatmat[:-1,:],optimize=True))
        hess = hesspt1 + hesspt2 + hesspt3
        
        return obj, gradients, jnp.conj(lambmat[0]), hess, jnp.conj(gradlamb[0])

    def a2objtraj(self, vhatmat):
        # Hamiltonian matrix 
        hhatmat = self.jkmat + vhatmat

        # eigendecomposition and compute propagator
        hatspec, hatstates = jnp.linalg.eigh(hhatmat)
        hatprop = hatstates @ jnp.diag(jnp.exp(-1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)
        hatpropH = hatstates @ jnp.diag(jnp.exp(1j*hatspec*self.dt)) @ jnp.conj(hatstates.T)

        #solve *forward* problem
        ahat = jnp.concatenate([jnp.expand_dims(self.jainit,0), jnp.zeros((self.nsteps, 2*self.nmax+1))])
        def forstep(j, loopamat):
            return loopamat.at[j+1].set( hatprop @ loopamat[j,:] )

        ahat = lax.fori_loop(0, self.nsteps, forstep, ahat)
        rhomat = self.vcorr(ahat, ahat, 'same') / jnp.sqrt(2 * self.biga)

        # compute only the objective function
        resid = rhomat - self.jbetamat
        obj = 0.5*jnp.real(jnp.sum(jnp.conj(resid)*resid))

        return obj, ahat

