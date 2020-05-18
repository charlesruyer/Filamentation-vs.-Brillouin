# -----------------------------------------------------------------------
# Brillouin - C. Ruyer - 12/2019
# -----------------------------------------------------------------------
#
# >>>>>> Analyse de l'XP omega EP de traverse de cavite
#
# >>>>>> Requirements
#   python2.7 with the following packages: numpy, matplotlib, pylab, scipy

# >>>>>> Advice: install ipython (historics of commands, better interface)
# >>>>>> First step: invoke python and load this file
#      $ ipython -i Luke_xp.py
#
# >>>>>> Second step: in the ipython shell, use the functions

from scipy.linalg import expm, inv

import scipy.special
from scipy import signal
from scipy.special import exp1
from scipy.special import  fresnel
import numpy as np
import numpy.matlib as ma
from scipy.interpolate import interp1d
import os.path, glob, re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from scipy.integrate import odeint
from scipy.integrate import complex_ode
from scipy.integrate import ode
from scipy.special import sici
from scipy.special import  lambertw
from scipy.special import  erf
from scipy.special import  erfi
from scipy import integrate
from scipy import optimize 
import pylab
import sys
#pylab.ion()

#Fonction plasma et ses derive a variable REELLE !
def Z_scalar(x):
    if np.abs(x)<=25:
        res = np.pi**0.5*np.exp(-x**2)*( 1j - erfi(x) ) 
    else:
        res=-1/x -1./x**3/2. +1j*np.sqrt(np.pi)*np.exp(-x**2)
    return res
def Z(x):
    if np.size(x)==1:
        return Z_scalar(x)
    else:
        res = 0*x+0j*x
        for i in range(len(x)):
            ##print('i= ',i, x[i], Z_scalar(x[i])
            res[i] = Z_scalar(x[i]) 
        return res 

def Zp(x):
    return -2*(1 + x*Z(x)) 

def Zp_scalar(x):
    return -2*(1 + x*Z_scalar(x))

def Zpp(x):
    return  -2*(Z(x) + x*Zp(x))

def Zppp(x):
    return  -2*( 2*Zp(x) + x*Zpp(x) )

def plot_alpha_kin(figure=1,Te=1.e3, ksk0max=0.02,k0=2*np.pi/0.35e-6,Z=[1.], A=[1.], nisne=[1.],nesnc=0.1):
    c=3e8
    #Ti=1000.
    Z=np.array(Z)
    vec=Z/Z
    #cs = 0.5*np.sqrt((Te+3*Ti)/511000./1836.)*c
    #vd=0.8*cs
    w0=k0*c/np.sqrt(1-nesnc)
    wpe2 = w0**2*nesnc
    ks = np.linspace(-ksk0max*k0,ksk0max*k0,1000)
    vp = (-0.5*np.abs(ks) *c**2/w0 -0*ks/np.abs(ks))
    #ak1 = alpha_kin(xie,xii,1)
    lde2   = Te/511000.*c / wpe2
    k2lde2 = ks**2 * lde2
    ak1 = alpha_kin(Te=Te, Ti=Te/1.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None,k2lde2=k2lde2) 
    ak3 = alpha_kin(Te=Te, Ti=Te/3.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None,k2lde2=k2lde2) 
    ak5 = alpha_kin(Te=Te, Ti=Te/5.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None,k2lde2=k2lde2) 

    ak1d2=np.squeeze(0.5*Fkin_Drake_generalise(Ti=Te/1.,Te=Te,Z=Z[0],A=A[0],ksx=ks,ksy=np.array([1e-10]),nesnc=nesnc,k0=2*np.pi/0.35e-6, figure=None).T)
    ak3d2=np.squeeze(0.5*Fkin_Drake_generalise(Ti=Te/3.,Te=Te,Z=Z[0],A=A[0],ksx=ks,ksy=np.array([1e-10]),nesnc=nesnc,k0=2*np.pi/0.35e-6, figure=None).T)
    ak5d2=np.squeeze(0.5*Fkin_Drake_generalise(Ti=Te/5.,Te=Te,Z=Z[0],A=A[0],ksx=ks,ksy=np.array([1e-10]),nesnc=nesnc,k0=2*np.pi/0.35e-6, figure=None).T)

    #print(ak1d2)
    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure,figsize=[7,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.19, right=0.92, top=0.9, bottom=0.16)
        l1,=plt.plot(ks/k0, np.real(ak1), 'k',linewidth=2 )
        l3,=plt.plot(ks/k0, np.real(ak3) , 'r',linewidth=2)
        l5,=plt.plot(ks/k0, np.real(ak5), 'b',linewidth=2 )
        l1d,=plt.plot(ks/k0, np.real(ak1d2), '--k',linewidth=2 )
        l3d,=plt.plot(ks/k0, np.real(ak3d2) , '--r',linewidth=2)
        l5d,=plt.plot(ks/k0, np.real(ak5d2), '--b',linewidth=2 )
        prop = fm.FontProperties(size=18)
        plt.legend([l1,l3,l5],['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=1, prop=prop,bbox_to_anchor=(1.1, 0.5))
        ax.set_xlabel("$k_s/k_0$")
        ax.set_ylabel("$\\Re(\\alpha_\\mathrm{kin})$")
        if ksk0max<0.025:
            ax.set_xticks([-0.02,-0.01,0,0.01,0.02]) 
        ax.set_xlim(-ksk0max,ksk0max)
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        plt.show()

        # plt.rcParams.update({'font.size': 20})
        # fig = plt.figure(figure,figsize=[7,5])
        # fig.clf()
        # ax = fig.add_subplot(1,1,1)
        # plt.subplots_adjust(left=0.19, right=0.92, top=0.9, bottom=0.16)
        # l1,=plt.plot(ks/k0, np.imag(ak1), 'k',linewidth=2 )
        # l3,=plt.plot(ks/k0, np.imag(ak3) , 'r',linewidth=2)
        # l5,=plt.plot(ks/k0, np.imag(ak5), 'b',linewidth=2 )
        # l1d,=plt.plot(ks/k0, np.imag(ak1d2), '--k',linewidth=2 )
        # l3d,=plt.plot(ks/k0, np.imag(ak3d2) , '--r',linewidth=2)
        # l5d,=plt.plot(ks/k0, np.imag(ak5d2), '--b',linewidth=2 )
        # prop = fm.FontProperties(size=18)
        # plt.legend([l1,l3,l5],['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=1, prop=prop,bbox_to_anchor=(1.1, 0.5))
        # ax.set_xlabel("$k_s/k_0$")
        # ax.set_ylabel("$\\Re(\\alpha_\\mathrm{kin})$")
        # if ksk0max<0.025:
        #     ax.set_xticks([-0.02,-0.01,0,0.01,0.02]) 
        # ax.set_xlim(-ksk0max,ksk0max)
        # #ax.set_xscale("log")
        # #ax.set_yscale("log")
        # plt.show()

# Genere la figure 1 de l article FSBS
def plot_alpha_kinfinal(figure=1):
    ksk0max=0.02
    k0=2*np.pi/0.35e-6
    Z=[1.]
    A=[1.]
    nisne=[1.] 
    ksk0max=2e-2
    c=3e8
    #Ti=1000.
    Z=np.array(Z)
    vec=Z/Z
    #cs = 0.5*np.sqrt((Te+3*Ti)/511000./1836.)*c
    #vd=0.8*cs
    w0=k0*c*np.sqrt(1.1)
    ks = np.linspace(-ksk0max*k0,ksk0max*k0,1000)
    vp = (-0.5*np.abs(ks) *c**2/w0 -0*ks/np.abs(ks))
    #ak1 = alpha_kin(xie,xii,1)
    Te=1e3
    ak11 = alpha_kin(Te=Te, Ti=Te/1.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None) 
    ak31 = alpha_kin(Te=Te, Ti=Te/3.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None) 
    ak51 = alpha_kin(Te=Te, Ti=Te/5.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None) 

    #ak1 = alpha_kin(xie,xii,1)
    Te=4.e3
    ak14 = alpha_kin(Te=Te, Ti=Te/1.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None) 
    ak34 = alpha_kin(Te=Te, Ti=Te/3.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None) 
    ak54 = alpha_kin(Te=Te, Ti=Te/5.*vec, Z=Z, A=A, nisne=nisne, vphi=vp,figure=None) 



    Te=1e3
    ksx= np.linspace(-0.014,0.014,300)*k0
    fkin=0.5*Fkin_Drake_generalise(Ti=Te/3.,Te=Te,Z=1.,A=1.,ksx=ksx,ksy=ksx,nesnc=0.1,k0=2*np.pi/0.35e-6, figure=None)


    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure,figsize=[7,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.22, right=0.89, top=0.9, bottom=0.16)
        l1,=plt.plot(ks/k0, np.real(ak11), 'k',linewidth=2 )
        l3,=plt.plot(ks/k0, np.real(ak31) , 'r',linewidth=2)
        l5,=plt.plot(ks/k0, np.real(ak51), 'b',linewidth=2 )
        l14,=plt.plot(ks/k0, np.real(ak14), '--k',linewidth=2 )
        l34,=plt.plot(ks/k0, np.real(ak34) , '--r',linewidth=2)
        l54,=plt.plot(ks/k0, np.real(ak54), '--b',linewidth=2 )
        prop = fm.FontProperties(size=18)
        first_legend = ax.legend([l1,l14],['$T_e=1$ keV','$T_e=4$ keV'],loc=1,prop=prop,bbox_to_anchor=(1.15, 0.35))#,numpoints=1)
        plt.gca().add_artist(first_legend)
        plt.legend([l1,l3,l5],['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=4, prop=prop,bbox_to_anchor=(1.15, 0.65))
        ax.set_ylabel("$\\Re(\\alpha_\\mathrm{kin})$")
        ax.set_xlabel("$k_s/k_0$")
        ax.set_xticks([-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04]) 
        ax.set_xlim(np.min(ks/k0),np.max(ks/k0))
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1,figsize=[7,5])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.2)
        data=(fkin.T)
        cf0 = plt.pcolor(ksx/k0 ,ksx/k0 , data,cmap =  plt.cm.RdBu)#, vmin =-1, vmax =1 )
        plt.colorbar(cf0)#, ticks=[ -1,0, 1])
        ax0.set_xlim(-0.01,0.01)
        ax0.set_ylim(-0.01,0.01)
        plt.axes().set_aspect('equal', 'datalim')
        ax0.set_xlabel("$k_{s,x}/k_0$")
        ax0.set_ylabel("$k_{s,y}/k_0$")
        fig.canvas.draw()

# Fonction de transfert cinetique de la force ponderomotrice
# Zp(xie)/2 * ( 1 - sum_i Zp(xi) ) /eps
def alpha_kin(Te, Ti, Z, A, nisne, vphi, k2lde2=0, ne=1,figure=None, is_chiperp = False):
    c=3e8
    if ne ==0 : # On neglige les e-
        Zpxie = -2+0j
    else: 
        xie = vphi /np.sqrt(2*Te/511000. )/c
        Zpxie = Zp(xie)
    ##print('Zp(xie) = ', Zpxie
    Xe= Zpxie * 1.
    sumXi = 0. + 0j
    for i in range(len(Ti)):
        xi = vphi /np.sqrt(2*Ti[i]/511000./1836/A[i] )/c
        sumXi += Zp(xi) * nisne[i] *Te* Z[i]**2/Ti[i]
        
        ##print('Zp(xii) = ', Zp(xi)
    # Si k2lde2 ==0  on suppose k2lde2 <<  1 (Longeur de Debaye electronique
    ak = -0.5*Zpxie *(k2lde2- sumXi) / (k2lde2 - Xe-sumXi)
    ##print(np.shape(np.real(ak)), np.shape(np.imag(ak))

    fluctuation=np.imag(1./ (+Xe+sumXi))
    if figure is None:
        return  ak
    else:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        a, = ax.plot(vphi/c,np.real(ak),'k', linewidth=2)              
        a, = ax.plot(vphi/c,np.imag(ak),'--k', linewidth=2)              
        ax.set_ylabel("$F_\mathrm{kin}$")
        ax.set_xlabel("$v_\phi/c$ ")
        #if ylog:
        #    ax.set_yscale("log")
        #if xlog:
        #    ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        a, = ax.plot(vphi/c,fluctuation,'k', linewidth=2)                
        ax.set_ylabel("$\\Im( 1./\\epsilon)$")
        ax.set_xlabel("$v_\phi/c$ ")
        #if ylog:
        #    ax.set_yscale("log")
        #if xlog:
        #    ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()




def gain_exact(k,x,g0tilde,Gamma0,D=1,te=1e3,ztesti=3,figure=1,nesnc=0.1, Z=[1.], A=[1.], nisne=[1.],f=6.5,k0=2*np.pi/0.35e-6,Nk=50000):
    l0=2*np.pi/k0
    km=k0/2./f
    c=3e8
    nc = 9.1e-31*8.85e-12/(1.6e-19)**2*k0**2*c**2
    print('nc = ',nc)
    ks=np.linspace(0,2*km,Nk)
    geom = 1.
    if D==2:
        geom = 2*np.pi*ks
    vphi =-0.5*np.abs(ks)*c/k0
    ak=np.real(alpha_kin(te,[Z[0]*te/ztesti], Z, A, nisne, (vphi), k2lde2=0, ne=1))
    A0=2*np.trapz(ak*geom,ks)/(2*np.pi)**D
    G=0*k
    xm= np.pi*k0/km**2 
    xint=np.linspace(xm,x,20000)
    g0=0*xint
    G0=0*xint
    for ix in range(len(xint)):
        g0[ix] = 2*np.trapz( -Gamma0*ak*np.sin(ks**2*xint[ix]/(4*k0) )*geom ,ks) /(2*np.pi)**D
        G0[ix] = np.trapz( g0[range(ix+1)], xint[range(ix+1)])
    for ik in range(len(k)):
        g=-A0*Gamma0*np.sin(k[ik]**2*xint/(4*k0)) *np.exp(G0)
        G[ik] = np.trapz( g, xint )

    G  *= np.exp(-G0[ len(xint)-1 ])
    Gap = plot_gain(k=k,x=np.array([x]),D=D,g0=g0tilde,Gamma0=Gamma0,te=te,figure=None,nesnc=nesnc,Z=Z,A=A,nisne=nisne,f=f,k0=k0,Nk=Nk)
    Gap = np.squeeze(Gap)

    Nm=int(np.floor(km**2*np.abs(x)/(2*np.pi*k0) +0.25))
    Gapp= []
    kp  = []
    for n in range(1,Nm+1):
        kn  = np.sqrt( 4*np.pi*k0/x*(2*n-0.5) ) 
        kp  = np.concatenate((kp,[ kn]))
        Gapp= np.concatenate((Gapp,[ Gap[np.argmin(np.abs(k-kn))]]))

    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)    
        th,   = ax.plot(k/km,G,'k', linewidth=2)   
        ap,   = ax.plot(k/km,Gap,'--k', linewidth=2)  
        app,  = ax.plot(kp/km , Gapp     , 'ok', linewidth=2,ms=10,markeredgecolor='black')

        prop = fm.FontProperties(size=18)
        #first_legend = ax.legend([th2,th3],['Aproximation, $D=1$','Aproximation, $D=2$'],loc=3,prop=prop)#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        plt.legend([th,ap,app],['Exact','Aproximation','$k=k_n$'],loc='best', prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$\\mathcal{I}$")
        ax.set_xlabel("$k/k_m$")
        #ax.set_xlim(-600,600)

        gM=np.max([np.max(G),np.max(Gap)])
        gm=np.min([np.min(G),np.min(Gap)])
        ax.set_ylim(1.1*gm,1.1*gM)
        ax.set_xlim(np.min(k/km),np.max(k/km))
        #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

def plot_gain(x,k,g0,Gamma0,D=1,te=1e3,ztesti=3,figure=1,nesnc=0.1, Z=[1.], A=[1.], nisne=[1.],f=6.5,k0=2*np.pi/0.35e-6,Nk=20000):
    l0=2*np.pi/k0
    km=k0/2./f
    c=3e8
    nc = 9.1e-31*8.85e-12/(1.6e-19)**2*k0**2*c**2
    print('nc = ',nc)
    ks=np.linspace(0,km,Nk)

    geom = 1.
    if D==2:
        geom = 2*np.pi*ks
    xc = np.pi*k0/km**2
    x=x[x>=xc]
    vphi =-0.5*np.abs(ks)*c/k0
    ak=np.real(alpha_kin(te,[Z[0]*te/ztesti], Z, A, nisne, (vphi), k2lde2=0, ne=1))
    A0=2*np.trapz(ak*geom,ks)/(2*np.pi)**D
    phi = np.arctan(ks**2/(4*k0)*g0)

    K,X = np.meshgrid(k,x)

    Phi = np.arctan(K**2/(4*k0)*g0)
    G= -A0*(np.cos(K**2*X/(4*k0)-Phi)-np.exp(-g0*X)*np.cos(K**2*xc/(4*k0)-Phi))/(np.sqrt((g0/Gamma0)**2+K**4/(16*k0**2*Gamma0**2)))#*np.exp(gt*(x-xc))

    if figure is None:
        return G.T
    else:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure,figsize=[7,5])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)   
        vm=np.max(np.abs(G))
        cf0 = plt.pcolor(km**2*x/k0,k/km,(G.T),cmap=plt.cm.RdBu,vmin=-vm,vmax=vm)#,cmap=plt.cm.gist_earth_r, vmin =0, vmax =np.max( np.log10(G.T)) )
        #ctmin,ctmax=np.floor(np.min(np.log10(G.T)))+1, np.floor(np.max(np.log10(G.T)))
        #nct=ctmax-ctmin+1
        #nr=nct
        #if nr>10:
        #    nr=np.floor(nct/2.)
        #print('ctmin, ctmax, nr = ', ctmin, ctmax,nr
        #ct=range(int(ctmin),int(ctmax+1))
        plt.colorbar(cf0)#, ticks=ct)
        ax0.set_xlim(np.min(g0),np.max(g0))
        ax0.set_ylim(np.min(te),np.max(te))
        #plt.axes().set_aspect('equal', 'datalim')
        ax0.set_ylabel("$k/k_m$")
        ax0.set_xlabel("$k_m^2x/k_0$")
        ax0.set_xlim(km**2*xc/k0,np.max(km**2*x/k0))
        ax0.set_ylim(np.min(k/km),np.max(k/km))
        fig.canvas.draw()
        plt.show()
        
def plot_sink2(figure=1):
    f=6.5
    k0=2*np.pi/0.35e-6
    km=k0/2./f
    x=np.linspace(0,2e-3,300)
    k=np.linspace(0,2*km,250)
    K,X = np.meshgrid(k,x)
    sin2 = -np.sin(K**2*X/(4*k0))
    sin2 = sin2.T
    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure,figsize=[7,5])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.2)
        cf0 = plt.pcolor(km**2*x/k0,k/km , sin2,cmap=plt.cm.RdBu, vmin =-1, vmax =1 )
        plt.colorbar(cf0, ticks=[-1,0,1])
        ax0.set_xlim(np.min(km**2*x/k0),np.max(km**2*x/k0))
        ax0.set_ylim(np.min(k/km),np.max(k/km))
        ax0.set_ylabel("$k/k_m$")
        ax0.set_xlabel("$k_m^2 x/k_0$")
        fig.canvas.draw()
        plt.show()

def int_alpha_kin_final(figure=1):
    x=np.linspace(-6e-3,6e-3,1000)
    sigma=100e-6
    ksmsk0=10e-2
    Te=1.e3
    Ti=[300.]
    Z=[1.]
    A=[1.]
    nisne=[1.]
    k2lde2=0
    ne=1
    f=6.5
    k0=2*np.pi/0.35e-6
    xlog=False
    ylog=False
       
    c=3e8
    km=k0/(2*f)
    kint=np.linspace(-ksmsk0,+ksmsk0,50000)*k0
    #kint=np.linspace(-2*km,+2*km,50000)
    F = 0*x + 0j*x
    vphi= -0.5*(np.abs(kint)/ k0)*c
    ak=np.real(alpha_kin(Te, Ti, Z, A, nisne, (vphi), k2lde2, ne))
    for i in range(len(x)):
        F[i] = np.trapz( np.exp(1j*kint**2/(2*k0)*x[i]) *ak , kint ) 
    #print(cF 
  
    ak0=np.trapz(ak , kint )
    ak2=np.trapz(ak *kint**2 , kint )
    ak4=np.trapz(ak  *kint**4, kint )
    Fa = ak0+1j*x*ak2/ (2*k0) -x**2*ak4 /(8*k0**2)

    if figure is not None:
    
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)     
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] = np.trapz( -np.sin(kint**2/(4*k0)*x[i]) *ak , kint )  /(2*np.pi)#*sigma
            ak, = ax.plot(km**2*x/k0,np.real(F)/km,col[c], linewidth=2)           
            #ki, = ax.plot(km**2*x/k0,np.imag(F),'--'+col[c], linewidth=2)   
            leg=leg+[ak]
            c+=1
        Te=4.e3 
        Tii =np.array([Te/1. ,Te/3.,Te/5.])
        c=0
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))
            for i in range(len(x)):
                F[i] = np.trapz( -np.sin(kint**2/(4*k0)*x[i]) *ak , kint )  /(2*np.pi)#*sigma
            akt, = ax.plot(km**2*x/k0,np.real(F)/km,'--'+col[c], linewidth=2)       
            c+=1    
        r, =plt.plot([],[],'k',   linewidth=2)
        i,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        first_legend = ax.legend([r ,i],['$T_e=1$ keV','$T_e=4$ keV'],loc=2,prop=prop)#,numpoints=1)
        plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=4, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g_0/\\Gamma_0/k_m$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        ax.set_xlim(-600,600)
        ax.set_ylim(-10e3/km,10e3/km)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)    
        Tii =np.array([1000./1. ,1000./3.,1000./5.])
        c=0
        col=['k','r','b']
        leg=[]
        vphi = vphi[kint>=0]
        kint = kint[kint>=0]
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            #ak=0.5*Fkin_Drake_generalise(Ti=Ti,Te=Te,Z=1.,A=1.,ksx=kint,ksy=0,nesnc=0.1,k0=2*np.pi/0.35e-6, figure=None)
            for i in range(len(x)):
                F[i] =   np.trapz(-2*np.pi*kint*np.sin(kint**2/(4*k0)*x[i]) *ak , kint )  /(2*np.pi)**2#*sigma**2
            ak,   = ax.plot(km**2*x/k0,np.real(F)/km**2,col[c], linewidth=2)       
            #aki, = ax.plot(km**2*x/k0,np.imag(F),'--'+col[c], linewidth=2)
            leg=leg+[ak]
            c+=1
        c=0
        Te=4.e3
        Tii =np.array([Te/1. ,Te/3.,Te/5.])
        for Ti in Tii:
            ak=np.real(alpha_kin(Te, [Ti], Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] =   np.trapz(-2*np.pi*kint*np.sin(kint**2/(4*k0)*x[i]) *ak , kint )  /(2*np.pi)**2#*sigma**2
            akt,   = ax.plot(km**2*x/k0,np.real(F)/km**2,'--'+col[c], linewidth=2)       
            c+=1
        r, =plt.plot([],[],'k',   linewidth=2)
        i,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        first_legend = ax.legend([r ,i],['$T_e=1$ kev','$T_e=4$ kev'],loc=2,prop=prop)#,numpoints=1)
        plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc=4, prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g_0/\\Gamma_0/k_m^2$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        ax.set_xlim(-600,600)
        ax.set_ylim(-1e9/km**2,1e9/km**2)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()



def int_alpha_kin(x,ksmsk0=2.e-2,Te=1.e3, Ti=[300.], Z=[1.], A=[1.], nisne=[1.], k2lde2=0, ne=1,f=6.5,k0=2*np.pi/0.35e-6,xlog=False, ylog=False, figure=1):    
    c=3e8
    km=k0/(2*f)
    kint=np.linspace(-ksmsk0,+ksmsk0,50000)*k0
    kintp=np.linspace(0,+ksmsk0,50000)*k0
    #kint=np.linspace(-2*km,+2*km,50000)
    F = 0*x + 0j*x    
    vphi= -0.5*(np.abs(kint)/ k0)*c
    vphip= -0.5*(np.abs(kintp)/ k0)*c
    Z=np.array(Z)
    testi=[1.,3.,5.]

    if figure is not None:
    
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)     
        #Tii =np.array([Te/1. ,Te/3.,Te/5.])
        Ti = Z/Z*Te
        c=0
        col=['k','r','b']
        leg=[]
        for testit in testi:
            ak=np.real(alpha_kin(Te, Ti/testit, Z, A, nisne, (vphi), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] = np.trapz( -np.sin(kint**2/(4*k0)*x[i]) *ak , kint )  /(2*np.pi)#*sigma
            ex, = ax.plot(km**2*x/k0,F/km,col[c], linewidth=2)    
            # dak=np.diff(ak)/(kint[1]-kint[0])
            # ddak=np.diff(dak)/(kint[1]-kint[0])
            # im=np.argmax( np.abs( dak ) )
            # dakm=np.abs(dak[im])
            # kcc = np.abs( 2*ak[im]/ddak[im] )**0.5
            # sq=0*x+0j*x
            # sq[x>0]=( 2*np.pi*k0/x[x>0] )**0.5
            # sq[x<0]=1j*( 2*np.pi*k0/np.abs(x[x<0]) )**0.5
            # #func=-sq*(sici( -( -kcc+k )/sq)[0] + sici(( kcc+k )/sq)[0] )
            # func=np.real(sq*2*fresnel(kcc /sq)[0])
            # A0 = np.mean(ak)
            # Fth = A0 *func /(2*np.pi)
            # th, = ax.plot(km**2*x/k0,Fth,'--'+col[c], linewidth=2)           

            leg=leg+[ex]
            c+=1    
        #e, =plt.plot([],[],'k',   linewidth=2)
        #th,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        #first_legend = ax.legend([e ,th],['Exact','Linearization'],loc=4,prop=prop)#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc='best', prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g_0/\\Gamma_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(-600,600)
        #ax.set_ylim(-0.6/sigma,0.6/sigma)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()


        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+2, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)    
        #Tii =np.array([1./1. ,1./3.,1/5.])*Te
        Ti = Z/Z*Te
        c=0
        col=['k','r','b']
        leg=[]
        for testit in testi:
            ak=np.real(alpha_kin(Te, Ti/testit, Z, A, nisne, (vphip), k2lde2, ne))+0
            for i in range(len(x)):
                F[i] =  np.trapz( -2*np.pi*kintp*np.sin(kintp**2/(4*k0)*x[i]) *ak , kintp )  /(2*np.pi)**2#*sigma**2
                
            ak,   = ax.plot(km**2*x/k0,np.real(F)/km**2,col[c], linewidth=2)       
            leg=leg+[ak]
            c+=1
        #e, =plt.plot([],[],'k',   linewidth=2)
        #th,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        #first_legend = ax.legend([e ,th],['Exact','Linearization'],loc=4,prop=prop)#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc='best', prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$g_0/\\Gamma_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(-600,600)
        #ax.set_ylim(-6/sigma**2,6/sigma**2)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+3, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)     
        #Tii =np.array([1./1. ,1./3.,1/5.])*Te
        Ti = Z/Z*Te
        c=0
        col=['k','r','b']
        leg=[]
        for testit in testi:
            ak=(alpha_kin(Te, Ti/testit, Z, A, nisne, (vphi), k2lde2, ne))+0
            A0=np.trapz(ak,kint)
            for i in range(len(x)):
                #F[i] = np.trapz( -np.sin(kint**2/(4*k0)*x[i]) *ak , kint )  /(2*np.pi)#*sigma
                F[i] = np.trapz( -np.exp(1j*kint**2/(4*k0)*x[i]) *ak , kint )  /(2*np.pi)#*sigma
            #ak, = ax.plot(km**2*x/k0,4*np.abs(A0/F)/(2*np.pi),col[c], linewidth=2)     

            ak, = ax.plot(km**2*x/k0,np.imag(F)/km,col[c], linewidth=2)          
            leg=leg+[ak]
            c+=1    
        #e, =plt.plot([],[],'k',   linewidth=2)
        #th,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        #first_legend = ax.legend([e ,th],['Exact','Linearization'],loc=4,prop=prop)#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc='best', prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$A_r\\Gamma_0/g_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(-600,600)
        #ax.set_ylim(-0.6/sigma,0.6/sigma)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()


        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+4, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)    
        #Tii =np.array([1./1. ,1./3.,1/5.])*Te
        Ti = Z/Z*Te
        c=0
        col=['k','r','b']
        leg=[]
        vphip = vphi[kint>=0]
        kintp = kint[kint>=0]
        for testit in testi:
            ak=(alpha_kin(Te, Ti/testit, Z, A, nisne, (vphip), k2lde2, ne))+0
            for i in range(len(x)):
                #F[i] =  np.trapz( -2*np.pi*kintp*np.sin(kintp**2/(4*k0)*x[i]) *ak , kintp )  /(2*np.pi)**2#*sigma**
                F[i] =  np.trapz( -2*np.pi*kintp*np.exp(1j*kintp**2/(4*k0)*x[i]) *ak , kintp )  /(2*np.pi)**2#*sigma**2
            #ak,   = ax.plot(km**2*x/k0,4*np.abs(A0/F)/(2*np.pi)**2,col[c], linewidth=2)
            ak, = ax.plot(km**2*x/k0,np.imag(F)/km**2,col[c], linewidth=2)     
            leg=leg+[ak]
            c+=1
        #e, =plt.plot([],[],'k',   linewidth=2)
        #th,=plt.plot([],[],'--k', linewidth=2)
        prop = fm.FontProperties(size=18)
        #first_legend = ax.legend([e ,th],['Exact','Linearization'],loc=4,prop=prop)#,numpoints=1)
        #plt.gca().add_artist(first_legend)
        plt.legend(leg,['$ZT_e/T_i=1$','$ZT_e/T_i=3$','$ZT_e/T_i=5$'],loc='best', prop=prop)#,bbox_to_anchor=(1.1, 0.5))     
        ax.set_ylabel("$A_r\\Gamma_0/g_0$")
        ax.set_xlabel("$k_m^2x/k_0$")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        #ax.set_xlim(-600,600)
        #ax.set_ylim(-6/sigma**2,6/sigma**2)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        fig.canvas.draw()
        plt.show()



##################################################################################"
#Calcul des matrices de susceptibilite Maxwellien avec derive suivant x et k qqc
# xi    = w/k/sqrt(2T/m) reel
# theta : angle de k avec x 
# vd    : derive normalise a la temperature
chixx =  lambda xi, theta, vd : -1 + np.cos(theta)**2*(1-xi**2*Zp_scalar(xi)) - (vd**2+np.sin(theta)**2)*Zp_scalar(xi)*0.5 -2*vd/2**0.5*np.cos(theta)*xi*Zp_scalar(xi)
chiyy =  lambda xi, theta, vd : -1 + np.sin(theta)**2*(1-xi**2*Zp_scalar(xi)) - np.cos(theta)**2*Zp_scalar(xi)*0.5
chixy =  lambda xi, theta, vd : -np.sin(theta)*np.cos(theta) *xi*(Z_scalar(xi)+xi*Zp_scalar(xi)) - np.sin(theta)*vd/2**0.5*xi*Zp_scalar(xi)
chiyx =  chixy  
Chi   =  lambda xi, theta, vd : np.matrix([[ chixx(xi,theta,vd) , chixy(xi,theta,vd) ],[chixy(xi,theta,vd), chiyy(xi,theta,vd) ]])
####################################################################################"

#Plot la partie cinetique de Drake generalise
#on se restrain a k dans le plan  x y
def Fkin_Drake_generalise(Ti, Te, Z, A,  ksx,ksy, nesnc,  vde=0,vdi=0, k0=2*np.pi/0.35e-6, figure=1,cscale='lin'):
    c=3e8
    w0 = k0*c/np.sqrt(1-nesnc)
    Id =  np.matrix([[1.,0.],[0.,1.]]) 
    wpse = nesnc**0.5*w0
    wpsi = np.sqrt(Z/1836./A)*wpse
    ve, vi = vde/c /np.sqrt(Te/511000.), vdi/c/np.sqrt(Ti/511000./A/1836.)
    #FSBS
    Ky, Kx = np.meshgrid(ksy,ksx) 
    theta  = np.arctan(Ky/Kx) +np.pi*(Kx<0)
    K = np.sqrt(Kx**2+Ky**2)

    #w = -0.5*K**2*c**2/w0
    w = w0*( 1- np.sqrt( 1+c**2/w0**2*( K**2 ) )   )
    #Calcul de la matrice D 
    ##print(  'shape : ',  np.shape(w/K), np.shape(vdi), np.shape(np.cos(theta))
    #print('ve, vi = ',ve, vi
    xii = 2**-0.5*(w/K  -vdi*np.cos(theta)*K/np.abs(K))/np.sqrt(Ti/511000./1836./A)/c
    xie = 2**-0.5*(w/K  -vde*np.cos(theta)*K/np.abs(K))/np.sqrt(Te/511000.)/c
    Fkin =0*Kx + 0j*Kx
    
    for ix in range(len(ksx)):
        #print("Calcul  : ",ix, " / ", len(ksx)
        for iy in range(len(ksy)):
            Chie, Chii = Chi(xie[ix,iy],theta[ix,iy],ve), Chi(xii[ix,iy],theta[ix,iy],vi)
            ##print('xi = ', xii[ix,iy]
            ##print('Xe = ',  Chie
            ##print('Xi = ',  Chii
            th = theta[ix,iy]

            vphi = w[ix,iy]/K[ix,iy]
            D  = Id*(1-c**2/vphi**2) + wpse**2/w[ix,iy]**2*Chie +wpsi**2/w[ix,iy]**2*Chii + c**2/vphi**2*np.matrix([[np.cos(th)**2,np.cos(th)*np.sin(th)],[np.cos(th)*np.sin(th),np.sin(th)**2]])
            Di = Id*(1-c**2/vphi**2) + 0                        +wpsi**2/w[ix,iy]**2*Chii + c**2/vphi**2*np.matrix([[np.cos(th)**2,np.cos(th)*np.sin(th)],[np.cos(th)*np.sin(th),np.sin(th)**2]])
            eparallel = np.matrix([[np.cos(th)], [np.sin(th)]])
            sol = (eparallel.T*( Di )*D.getI()*eparallel)
            Fkin[ix,iy] =  -Zp_scalar(xie[ix,iy])*sol[0,0]
            
            vphi = -w[ix,iy]/K[ix,iy]
            D  = Id*(1-c**2/vphi**2) + wpse**2/w[ix,iy]**2*Chie +wpsi**2/w[ix,iy]**2*Chii + c**2/vphi**2*np.matrix([[np.cos(th)**2,np.cos(th)*np.sin(th)],[np.cos(th)*np.sin(th),np.sin(th)**2]])
            Di = Id*(1-c**2/vphi**2) + 0                        +wpsi**2/w[ix,iy]**2*Chii + c**2/vphi**2*np.matrix([[np.cos(th)**2,np.cos(th)*np.sin(th)],[np.cos(th)*np.sin(th),np.sin(th)**2]])
            eparallel = np.matrix([[np.cos(th)], [np.sin(th)]])
            sol = eparallel.T*( Di )*D.getI()*eparallel
            Fkin[ix,iy]+=  -0.25*Zp_scalar(xie[ix,iy])*sol[0,0]

    Fkin = np.real(Fkin)
    if figure is None:
        return Fkin
    else:
        print("Plot !")
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[7,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.2)
        if cscale=='log':
            data=np.log10(Fkin.T)
            dM = np.max(np.abs(data))
            dm = dM-3
        else:
            a=(Fkin.T)
            dM = np.max(np.abs(data))
            dm = 0
            cf0 = plt.pcolor(ksx/k0 ,ksy/k0 , data, vmin =-dM, vmax =dM ,cmap =  plt.cm.RdBu)
            plt.colorbar(cf0)#, ticks=[ 0, cmax/2., cmax])
            #plt.axes().set_aspect('equal', 'datalim')
            ax0.set_xlabel("$k_x/k_0$")
            ax0.set_ylabel("$k_y/k_0$")
            fig.canvas.draw()
            plt.show()




def Fkin(xie,xii,ztesti):
	sol = -0.5*Zp(xie)*Zp(xii) / (Zp(xii)+Zp(xie)/ztesti)
	return sol

# Convolution  de Im(Fkin exp(iwt)) par la forme du laser
def convolve_alphakin(Te,Ti,kskm,t,k0=2*np.pi/0.35e-6,Zi=1,A=1,f=6.5,figure=1, sigma = 1e10,nu=0.1):
    c  = 3e8
    km=k0/2./f
    k=kskm*km
    dk = (k[1]-k[0] )
    w0 = k0*c
    kk,tt = np.meshgrid(k,t)
    wplus = -0.5*kk**2*c**2/w0
    xii   = np.sqrt(1836.*511.e3/Ti/2.)*wplus/np.abs(kk)/c
    xie   = np.sqrt(511.e3/Te/2.)*wplus/np.abs(kk)/c
    cs=1./np.sqrt(1836.*511.e3/Ti)*(Zi*Te/Ti+1)**0.5*c
    ##print(xii
    ##print(xie
    def ZZ(x):
        res= 1j*np.sqrt(np.pi)*np.exp(-x**2)-1./x
        crit=np.abs(x)<20
        res[crit]=np.exp(-x[crit]**2)*np.pi**0.5*(1j-erfi(x[crit]))
        return res
    def ZZp(x):
        return -2*(1+x*Z(x))
    def alpha_kin(xie,xii,ztesti):
        return -0.5*ZZp(xie)*ZZp(xii) / (ZZp(xii)+ZZp(xie)/ztesti)
    akin  = -Fkin(xie,xii,Zi*Te/Ti) #* (np.abs(k)>kmin)
    expit = np.exp(1j*wplus*tt)
    if sigma<1e10:
        env =0*kk
        kp=np.linspace(-km,km,100)
        #for i in range(len(kp)):
        #	env += 2*sigma*np.sinc(sigma*(k-2*kp[i])/np.pi) / len(kp)
        #	H = env
        H = -2*(sici( (kk-2*km)*sigma)[0] -sici((kk+2*km)*sigma)[0]) /2./km
    else:
        env=1./2./km
        H = (np.abs(kk)<2*km) *env
    
    cv=0*tt
    #print(np.shape(cv)
    cv2=0*tt
    fki=0*tt
    fkr=0*tt
    for it in range(len(t)):
        fki[it,:] = (1-np.exp(-nu*np.abs(kk[it,:]*cs*tt[it,:]))*np.cos(kk[it,:]*cs*tt[it,:]))*np.imag(akin[it,:]*expit[it,:])
        fkr[it,:] = (1-np.cos(kk[it,:]*cs*tt[it,:]))*np.real(akin[it,:]*expit[it,:])
        cv[it,:] = np.convolve(H[it,:],fkr[it,:],mode='same')*dk
        cv2[it,:] = np.trapz(fkr[it,:]*H[it,:],k)*H[it,:]*2*km 
    
    #print(np.shape(cv)
    cvm = np.mean(cv *(np.abs(kk)<2*km),axis=1)
    #print(np.shape(cv)
    sk=1
    #print(np.shape(cv)
    if len(k)>1000:
        sk = int(len(k)/500.) 
    #print(np.shape(cv)
    mesh = np.meshgrid(  range(0,len(t),1), range(0,len(k),sk ) )
    #print(np.shape(cv)
    cvplot = cv[mesh]
    kplot = k[ range(0,len(k),sk )]/km
    indexmin = np.argmin(np.abs(k+0.1*km))
    indexmax = np.argmin(np.abs(k-0.1*km))
    mesh = np.meshgrid(  range(0,len(t),1),  range(indexmin,indexmax,1 )   )
    fkplot = fki[mesh]
    kplotfk= k[  range(indexmin,indexmax,1 )  ]/km
    #print(np.shape(fkplot) , np.shape(kplotfk)
    ##print(cv
    
    fig = plt.figure(figure)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.pcolor(t*1e12,kplot,cvplot)
    plt.colorbar()
    ax.set_xlabel("$t$ ps")
    ax.set_ylabel("$k/k_m$")
    #ax.set_xlim(np.min(x),np.max(x))
    #ax.set_ylim(np.min(y), np.max(y))
    fig.canvas.draw()
    plt.show()
    
    fig = plt.figure(figure+1)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.pcolor(t*1e12,kplotfk,fkplot)
    plt.colorbar()
    ax.set_xlabel("$t$ ps")
    ax.set_ylabel("$k/k_m$")    
    #ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(-0.1, 0.1)
    fig.canvas.draw()
    plt.show()

    fig = plt.figure(figure+2)
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.plot(t*1e12,cvm)
    ax.set_xlabel("$t$ ps")
    ax.set_ylabel("$G_k$")
    #ax.set_xlim(np.min(x),np.max(x))
    #ax.set_ylim(np.min(y), np.max(y))  
    fig.canvas.draw()
    plt.show()




#Plot comparaison FFT RCF et alphakin
#fileRCF="/ccc/cont001/home/f8/ruyer4s/XP_Omega_EP/Lineout_26325H8.txt"
#pour le C5H12 : nhsne=1./(6*5./12. + 1.) et ncsne = 5/12* nesne
def comp_RCF_alphakin(Te,Ti,Z,A,nisne,fileRCF,figure=1,ylog=True,xlog=True,facteur_comp=1):
    c=3e8
    k0=2*np.pi/0.35e-6
    w0 = k0*c
    y, dose = np.loadtxt(fileRCF,unpack=True)
    dose0 = dose - np.mean(dose)
    y=y*1e-6
        
    dosefft = np.fft.fftshift(np.fft.fft( (dose0) ) ) *(y[1]-y[0])    
    dk = 2*np.pi/(np.max(y)-np.min(y))
    kydose = dk*np.linspace(-int(len(y)/2),int(len(y)/2), len(y)+1)
    kydose = kydose[np.abs(kydose)>0]
    #print('RCF : '
    #print('kmax / k0 = ',np.max(kydose/k0)
    #print('kmin / k0 = ',np.min(np.abs(kydose/k0))

    ks = np.linspace(-np.max(kydose),np.max(kydose),1000)
    vphi = 0.5*ks*c**2/w0
    F_kin=np.imag(alpha_kin(Te=Te, Ti=Ti, Z=Z, A=A, nisne=nisne, vphi=vphi, k2lde2=0, ne=1))
        
    ne=0.05*9e27
    Ep=7.5*1e6
    Lz=1200e-6
    dnsn=0.02 #I= 3.10^14 W/cm^2 
    facteur_rcf=Te/(2*Ep)*500e-6 * dnsn* ks * facteur_comp
    #facteur_rcf=   facteur_comp
    
    r=taux_collisionnel(masse=[1., np.mean(A)*1836.],charge=[-1., np.mean(Z)],dens=[ne/1e+6, ne/1e+6/np.mean(Z)],temp=[Te, np.mean(Ti)],vd=[0., 0.])
    loglamb = r['log_coul']
    #print('Log( Lambda ) = ' , loglamb   
    lei = r['lmfp']*1e-2 
    Ak = 2 * (0.5+ 0.074/(np.abs(ks)*lei) + 0.88*np.mean(Z)**(5./7.) /(np.abs(ks)*lei)**(5./7.) + 2.54*np.mean(Z) /( 1.+5.5*(np.abs(ks)*lei)**2 ) ) 


    if figure is not None:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        ak, = ax.plot(ks/k0,facteur_rcf*Ak*np.abs(F_kin),'k', linewidth=2)              
        ak, = ax.plot(kydose/k0,np.abs(dosefft),'--k', linewidth=2)              
        #prop = fm.FontProperties(size=15)
        #ax.legend([h13,h14,h15],['$10^{13}$ W/cm$^{-2}$','$10^{14}$ W/cm$^{-2}$', '$10^{15}$ W/cm$^{-2}$' ], loc=9, bbox_to_anchor=(0.6, 0.35),prop = prop)#, borderaxespad=0.)
        ax.set_ylabel("FFT(dose)")
        ax.set_xlabel("$k_y/k_0$ ")
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()

def betak(x, ZTesTi):
    res = np.imag(Zp(x)/(Zp(x)-2./ZTesTi))
    return res
def betakr(x, ZTesTi, zsa = 1):
    xe = x *np.sqrt( zsa / 1836./ ZTesTi )
    res = np.real( Zp(x)/(Zp(x) + Zp(xe)/ZTesTi))
    return res

def d_betak(x, ZTesTi):
    res = -2./ZTesTi * np.imag( Zpp(x) / (Zp(x)-2./ZTesTi)**2 )
    return res
def dd_betak(x, ZTesTi):
    res = -2./ZTesTi* np.imag(( (Zp(x)-2./ZTesTi)*Zppp(x)-2*Zpp(x)**2)/(Zp(x)-2./ZTesTi)**3)
    return res

#Calcul le max de alpha kin et la largeur du pic
def Deltak_ksm_FSBS(ztesti,figure=None,f=6.5,k0=2*np.pi/0.35e-6):
    xim = 0*ztesti
    Dxi = 0*ztesti
    betakrr = 0*ztesti
    xit =np.logspace(-3,0.5,100000) 
    for i in range(len(ztesti)):
        dbeta = d_betak(xit,ztesti[i])
        xim[i] = xit[np.argmin(np.abs(dbeta))]
        Dxit = 0#( np.sqrt(-betak(xim[i],ztesti[i])/dd_betak(xim[i],ztesti[i]) ) )
        #print(-betak(xim[i],ztesti[i]), dd_betak(xim[i],ztesti[i])
        Dxi[i] = np.real(Dxit)
        #print('Precision a ZTe/Ti = ',ztesti[i],' est de ', np.imag(Dxit)/np.real(Dxit) 
        betakrr[i] = np.trapz(betakr(xit, ztesti[i]), xit )
    G=4.*4.*2**0.5/3. * xim*Dxi*betak(xim,ztesti) 
    
    #Deuxieme calcul : int int Im(betak) 
    def int_betak(x,u):
        if np.size(x)==1:
            x=np.array([x])
        sol=0*x
        for l in range(len(x)):
            xxi = np.linspace(np.max([-10,-10+x[l]]),np.min([5,5+x[l]]),1000)
            sol[l] = np.trapz(betak(xxi,u),xxi)
        return np.squeeze( sol )
    def intint_betak(x,u):   
        if np.size(x)==1:
            x=np.array([x])    
        sol=0*x
        for j in range(len(x)):
            xxi = np.linspace(np.max([-10,-10+x[j]]),np.min([5,5+x[j]]),1000)
            #xxi = np.linspace(-10+x[i],10+x[i],10000)
            sol[j] = np.trapz(int_betak(xxi,u),xxi)
        return np.squeeze(sol )
    #G2=0*ztesti
    #x = np.linspace(0,5,100)
    #for i in range(len(ztesti)):
    #    #print("Calcul 2, i = ", i," up to ", len(ztesti)
    #    G2[i] = np.max( intint_betak(x,ztesti[i]) )
    #    #print('G2 = ', G2[i]
        
    
    
    if figure is None:
        return {'ztesti':ztesti, 'xim':xim, 'Dxi':Dxi}
    else:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        x, = ax.plot(ztesti,xim,'k', linewidth=2)              
        dx, = ax.plot(ztesti,Dxi,'--k', linewidth=2)            	
        prop = fm.FontProperties(size=15)
        ax.legend([x,dx],['$x_\mathrm{max}$','$\Delta x$']
                  ,loc='best', #bbox_to_anchor=(1.35, 0.7), borderaxespad=0., 
                  prop=prop)
  
        #ax.set_ylabel("$\xi_\mathrm{max}$, $\Delta \xi$")
        ax.set_xlabel("$ZT_e/T_i$ ")
        #if ylog:
        ax.set_yscale("log")
        #if xlog:
        ax.set_xscale("log")
        ax.set_xlim(np.min(ztesti),np.max(ztesti))
        fig.canvas.draw()
        plt.show()
 
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        ak, = ax.plot(ztesti,G,'k', linewidth=2)                 
        #ak, = ax.plot(ztesti,G2,'--k', linewidth=2)              

        #ax.set_ylabel("$\xi_\mathrm{max}$, $\Delta \xi$")
        ax.set_xlabel("$ZT_e/T_i$ ")
        #if ylog:
        ax.set_yscale("log")
        #if xlog:
        ax.set_xscale("log")
        ax.set_xlim(np.min(ztesti),np.max(ztesti))
        fig.canvas.draw()
        plt.show()
 
 
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+2, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        ak, = ax.plot(ztesti,betakrr,'k', linewidth=2)                 
        #ak, = ax.plot(ztesti,G2,'--k', linewidth=2)              

        #ax.set_ylabel("$\xi_\mathrm{max}$, $\Delta \xi$")
        ax.set_xlabel("$ZT_e/T_i$ ")
        #if ylog:
        ax.set_yscale("log")
        #if xlog:
        ax.set_xscale("log")
        ax.set_xlim(np.min(ztesti),np.max(ztesti))
        fig.canvas.draw()
        plt.show()
 

def taux_collisionnel(masse=None,charge=None,dens=None,temp=None,vd=None,*aqrgs,**kwargs):
    #% Frequences de collision de Spitzer
    #% Ref. A. Decoster, Modeling of collisions (1998)
    #%
    #% Input : [beam, background]
    #% . masse(1:2)  : masse/me
    #% . charge(1:2) : charge/qe
    #% . dens(1:2)   : densite (cm^-3)
    #% . temp(1:2)   : temperature (eV)
    #% . vde(1:2)    : vitesses de derive (cm/s)
    #% Output :
    #% . nu_imp  : momentum transfer frequency (1/s)
    #% . nu_ener : Energy transfer frequency (1/s)
    #% . lambda  : mean-free-path (cm)
    masse = np.array(masse)
    charge = np.array(charge)
    dens = np.array(dens)
    temp = np.array(temp)
    vd = np.array(vd)
    
    #varargin = cellarray(args)
    #nargin = 5-[masse,charge,dens,temp,vd].count(None)+len(args)
    
    vt1=30000000000.0 * (temp[0] / (masse[0] * 511000.0)) ** 0.5
    vd1=np.abs(vd[0])
    if all(masse == 1):
        ##print('All electrons'
        if temp[1] < 10:
            log_coul=23 - np.log(dens[1] ** 0.5 * temp[1] ** (- 1.5))
        else:
            log_coul=24 - np.log(dens[1] ** 0.5 * temp[1] ** (- 1))
        ##print(log_coul, temp
    else:
        if any(masse == 1.):
            ##print('electron-ion'
            if masse[0] ==1.:
                ielec=0
                ##print('indice electron: ',ielec
                iion=1
                ##print('indice ion: ',iion
            else:
                ielec=1
                ##print('indice electron: ',ielec
                iion=0
                ##print('indice ion: ',iion
            
            
            if (temp[iion] / masse[iion] < temp[ielec]) and (temp[ielec] < 10 * charge[iion] ** 2):
                log_coul=23 - np.log(dens[ielec] ** 0.5 * charge[iion] * temp[ielec] ** (- 1.5))
            else:
                if 10 * charge[iion] ** 2 < temp[ielec]:
                    log_coul=24 - np.log(dens[ielec] ** 0.5 * temp[ielec] ** (- 1))
                else:
                    if temp[ielec] < temp[iion] * charge[iion] / masse[iion]:
                        mu=masse[iion]/1836.
                        log_coul=30 - np.log(dens[iion] ** 0.5 * temp[iion] **(-1.5) * charge[iion] ** 2 / mu)
                    else:
                        print( 'No Coulombien logarithm from Lee and Moore')
                        return  {"nup":None,"nuk":None,"log_coul":None,"lmfp":None}
            	##print('Log Coulombien: ',log_coul
        else:
            log_coul=23 - np.log(charge[0] * charge[1] * (masse[0] + masse[1]) * (dens[0] * charge[0] ** 2 / temp[0] + dens[1] * charge[1] ** 2 / temp[1]) ** 0.5 / (masse[0] * temp[1] + masse[1] * temp[0]))

    qe=4.8032e-10
    temp=1.6022e-19 * 10000000.0 * temp
    masse=9.1094e-28 * masse
    m12=masse[0] * masse[1] / (masse[0] + masse[1])
    nu_imp=(4. / 3.) * (2 * np.pi) ** 0.5 * dens[1] * charge[0] ** 2 * charge[1] ** 2 * qe ** 4 * log_coul / (masse[0] * m12 * (temp[0] / masse[0] + temp[1] / masse[1] + (vd[0] - vd[1]) ** 2) ** 1.5)
    nu_ener=(8. / 3.) * (2 * np.pi) ** 0.5 * dens[1] * charge[0] ** 2 * charge[1] ** 2 * qe ** 4 * log_coul / (masse[0] * masse[1] * (temp[0] / masse[0] + temp[1] / masse[1] + (vd[0] - vd[1]) ** 2) ** 1.5)

    _lambda=np.max([vt1,vd1]) / nu_imp
    ##print('nu_imp = ',nu_imp,' Hz'
    ##print('nu_ener = ',nu_ener,' Hz'
    ##print('tau_imp = ',1/nu_imp,' s'
    ##print('tau_ener = ',1/nu_ener,' s', dens, log_coul,masse
    ##print('log_coul = ',log_coul
    ##print('Mean-free-path: ',_lambda,' cm'
    result = {"nup":nu_imp,"nuk":nu_ener,"log_coul":log_coul,"lmfp":_lambda}
    return result


def dispe_filam_kin(Te=1.e3,Ti=[300.],Z=[1.],A=[1.],nesnc=0.1,I0=3e14,k0=2*np.pi/0.35e-6,nisne=None,figure=1,gmax=None,kmax=None,gticks=None,kticks=None):
    #dispe_filam_kin(Te=700.,Ti=[500,500.],Z=[1.,6.],A=[1.,12.],nisne=[1./7.,1./7.],gticks=[0,1e-3,2e-3],kticks=[0,1e-3,2e-3,3e-3],gmax=2e-3,kmax=3e-3)
    #dispe_filam_kin(Te=1000.,Ti=[300],Z=[1.],A=[1.],gmax=2.5e-4,kmax=5.5e-4,gticks=[0,1e-4,2e-4],kticks=[0,1e-4,2e-4,3e-4,5e-4])
    c=3e8
    nc=k0**2*c**2*8.85e-12*9.1e-31/(1.6e-19)**2
    print('nc = ',nc, ' m^-3')
    if nisne is None:
        nisne = [1./np.float(np.array(Z))]
        print('nisne = ',nisne)
    dnsn = I0*1e4/(nc*c*Te*1.6e-19)
    print('dnsn = ',dnsn)
    n=np.sqrt(1-nesnc)
    Tim, Am, Zm = np.mean(np.array(Ti)),np.mean(np.array(A)),np.mean(np.array(Z))
    cs = np.sqrt((Zm*Te+3*Tim)/(1836.*511000.*Am ))*c
    k = np.linspace(0,3*k0*dnsn**0.5,120)
    m=0
    for i in range(len(Ti)):
        m+=Z[i]**2*nisne[i]*Te/Ti[i]
    alphak = m/(1.+m)
    uf2 =0.25 *( 0.5*nesnc*dnsn/n**2*alphak - k**2/k0**2)
    ik = uf2>0

    kf,uf =k[ik], np.sqrt(uf2[ik])
    print('len uf = ',len(uf))
    u2 = 1+ 0.5/uf2 *(1.-np.sqrt(1+4*uf2))
    ik=u2>0
    ku,u = k[ik],np.sqrt(u2[ik])
    print('len u = ',len(u))

    ne=nesnc*nc/1e6
    print([1.,A[0]*1836.],[-1.,Z[0]],[ne,ne/Z[0]],[Te,Ti[0]])
    r=taux_collisionnel(masse=[1.,A[0]*1836.],charge=[-1.,Z[0]],dens=[ne,ne*nisne[0]],temp=[Te,Ti[0]],vd=[0,0])
    print(r['lmfp'])
    lmfp=r['lmfp']*1e-2
    iak=0
    if len(Z)>1:
        for i in range(1,len(Z)):
            r=taux_collisionnel(masse=[1.,A[i]*1836.],charge=[-1.,Z[i]],dens=[ne,ne*nisne[i]],temp=[Te,Ti[i]],vd=[0,0])
            print('lmfp, i = ',i, '   : ',r['lmfp'])
            if lmfp  > r['lmfp']*1e-2:
                lmfp = r['lmfp']*1e-2 
                iak  = i

    print('lmfp = ',lmfp,' m')
    x=lmfp*Z[i]**0.5*k
    Ak =( 0.5 +Z[i]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )
    ufa2 = 0.25*( nesnc*dnsn/n**2*alphak*Ak - k**2/k0**2 )
    ik = ufa2>0
    kfa,ufa =k[ik], np.sqrt(ufa2[ik])
    print('len ufa = ',len(ufa))
    ua2 = 1+ 0.5/ufa2 *(1.-np.sqrt(1+4*ufa2))
    ik=ua2>0
    kua,ua = k[ik],np.sqrt(ua2[ik])
    print('len ua = ',len(ua))

    #Brillouin avant 

    k=np.linspace(0,1.5*np.max(kua),120)
    x=lmfp*Z[i]**0.5*k
    Ak =( 0.5 +Z[i]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )

    #g0= np.sqrt(np.pi) *(cs/c)**3*(np.exp(-Am*1836*9.1e-31*cs**2/(2*Tim*1.6e-19))/(2*Tim/511000.)**1.5 +Am*1836./Zm/(2*Te/511000.)**1.5)
    vi=np.sqrt(Tim/(Am*1836*511000.))*c
    g0= np.sqrt(np.pi/8) *(cs/vi)**3*(np.exp(-0.5*(cs/vi)**2 ) +np.sqrt(Zm/(Am*1836.)) *(Zm*Te/Tim)**-1.5)

    print('g_0 = ', g0)

    vphi = np.linspace(0.*cs, 1.2*cs,100)
    V, K=np.meshgrid(vphi,k)
    alphakold=alphak
    print('alpha[0] = ',alphak )
    alphak = alpha_kin(Te,Ti, Z, A, nisne, (vphi), k2lde2=0, ne=1)
    alphaf = 1./( 1-2*1j*g0*vphi/cs - (vphi/cs)**2  )
    print('alpha[0] = ',alphak[0] )
    print(np.shape(alphak), np.shape(Ak))
    #print('alphak = ',alphak
    Up = 0*V +0j*V
    Ufp = 0*V +0j*V
    Um = 0*V +0j*V
    Ufm = 0*V +0j*V
    for iv in range(len(vphi)):
        for ik in range(len(k)): 
            #cinetique
            a=1
            b=-2*vphi[iv]/n/c
            C=(vphi[iv]/n/c)**2-0.25*(k[ik]**2/k0**2-nesnc*dnsn/n**2*alphak[iv]*Ak[ik] )
            delta = b**2-4*a*C
            thd=np.arctan(np.imag(delta)/np.real(delta))+np.pi*(np.real(delta)<0)
            rd = np.abs(delta)
            Up[ik,iv] =  (-b+rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            Um[ik,iv] =  (-b-rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            
            #Fluide
            a=1
            b=-2*vphi[iv]/n/c
            C=(vphi[iv]/n/c)**2-0.25*(k[ik]**2/k0**2-nesnc*dnsn/n**2*alphaf[iv]*Ak[ik] )
            delta = b**2-4*a*C
            thd=np.arctan(np.imag(delta)/np.real(delta))+np.pi*(np.real(delta)<0)
            rd = np.abs(delta)
            Ufp[ik,iv] =  (-b+rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            Ufm[ik,iv] =  (-b-rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            
    U= Up*(np.imag(Up)>0) + Um*(np.imag(Um)>0)
    Uf= Ufp*(np.imag(Ufp)>0) + Ufm*(np.imag(Ufm)>0)
            
    gamma = K*np.imag(U)
    kpara = K*np.real(U)
    gammaf = K*np.imag(Uf)
    kparaf = K*np.real(Uf)
    #np.arctan2(np.real(U2),np.imag(U2))/2.
    #uabs = np.sqrt(np.abs(U2))
    #gamma = K*uabs*np.cos(uth)
    #kpara = -K*uabs*np.sin(uth)

    #mesh  = gamma>0
    #gamma *= mesh
    #kpara *= mesh
    for iv in range(len(vphi)):
        for ik in range(len(k)):
            if np.isnan(gamma[ik,iv]) or np.isnan(kpara[ik,iv]):
                gamma[ik,:iv] = 0
                kpara[ik,:iv] = 0

            if np.isnan(gammaf[ik,iv]) or np.isnan(kparaf[ik,iv]):
                gammaf[ik,:iv] = 0
                kparaf[ik,:iv] = 0

    #gmax=np.max([np.max(gamma),np.max(gammaf)])/k0
    #kmax=np.max([np.max(kpara),np.max(kparaf)])/k0
    #print('gamma/k0 = ',gamma /k0
    #print('kpara/k0 = ',kpara/k0
    if figure is not None:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.2)
        plt.plot(ku/k0 , ku*u/k0,'k',linewidth=2)
        plt.plot(kf/k0 , kf*uf/k0,'--k',linewidth=2)
        plt.plot(kua/k0 , kua*ua/k0,'r',linewidth=2)
        plt.plot(kfa/k0 , kfa*ufa/k0,'--r',linewidth=2)
        plt.plot(k/k0,np.squeeze(gamma[:,0])/k0,'g',linewidth=2)
        ax0.set_ylabel("$\\Gamma/k_0$")
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=gamma.T/k0
        #data = np.log10(gamma)
        if gmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =0, vmax =gmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if gticks is not None:
            plt.colorbar(cf0, format='%.0e', ticks=gticks)
        else:
            plt.colorbar(cf0, format='%.0e')
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+2,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=-kpara.T/k0
        #data = np.log10(gamma)
        if kmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =0, vmax =kmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if kticks is not None:
            plt.colorbar(cf0, format='%.0e', ticks=kticks)
        else:
            plt.colorbar(cf0, format='%.0e')
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+3,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=gammaf.T/k0
        #data = np.log10(gamma)
        if gmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =0, vmax =gmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if gticks is not None:
            plt.colorbar(cf0, format='%.0e', ticks=gticks)
        else:
            plt.colorbar(cf0, format='%.0e')
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+4,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=-kparaf.T/k0
        if kmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =0, vmax =kmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if kticks is not None:
            plt.colorbar(cf0, format='%.0e', ticks=kticks)
        else:
            plt.colorbar(cf0, format='%.0e')
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
def dispe_filam_kinRPP(Te=1.e3,Ti=[300.],Z=[1.],A=[1.],nesnc=0.1,I0=3e14,f=8.,k0=2*np.pi/0.35e-6,nisne=None,isnonlocal=True,k=None,vpscs=None,figure=1,gmax=None,kmax=None,gticks=None,kticks=None):
    #dispe_filam_kin(Te=700.,Ti=[500,500.],Z=[1.,6.],A=[1.,12.],nisne=[1./7.,1./7.],gmax=5.e-2,kmax=0.1,gticks=[0,0.01,0.02,0.03,0.04,0.05],kticks=[-0.1,0,0.1])
    #dispe_filam_kin(Te=1000.,Ti=[300],Z=[1.],A=[1.],gmax=1.3e-2,kmax=0.1,gticks=[0,0.01],kticks=[-0.1,0,0.1])
    c=3e8
    km=k0/(2*f)
    nc=k0**2*c**2*8.85e-12*9.1e-31/(1.6e-19)**2
    print('nc = ',nc, ' m^-3')
    if nisne is None:
        nisne = [1./np.float(np.array(Z))]
        print('nisne = ',nisne)
    dnsn = I0*1e4/(nc*c*Te*1.6e-19)
    print('dnsn = ',dnsn)
    n=np.sqrt(1-nesnc)
    w0=k0*c/n
    Tim, Am, Zm = np.mean(np.array(Ti)),np.mean(np.array(A)),np.mean(np.array(Z))
    cs = np.sqrt((Zm*Te+3*Tim)/(1836.*511000.*Am ))*c
    m=0
    for i in range(len(Ti)):
        m+=Z[i]**2*nisne[i]*Te/Ti[i]
    alphak = m/(1.+m)
    
    ne=nesnc*nc/1e6
    print([1.,A[0]*1836.],[-1.,Z[0]],[ne,ne/Z[0]],[Te,Ti[0]])
    r=taux_collisionnel(masse=[1.,A[0]*1836.],charge=[-1.,Z[0]],dens=[ne,ne*nisne[0]],temp=[Te,Ti[0]],vd=[0,0])
    print(r['lmfp'])
    lmfp=r['lmfp']*1e-2
    iak=0
    if len(Z)>1:
        for i in range(1,len(Z)):
            r=taux_collisionnel(masse=[1.,A[i]*1836.],charge=[-1.,Z[i]],dens=[ne,ne*nisne[i]],temp=[Te,Ti[i]],vd=[0,0])
            print('lmfp, i = ',i, '   : ',r['lmfp'])
            if lmfp  > r['lmfp']*1e-2:
                lmfp = r['lmfp']*1e-2 
                iak  = i

    print('lmfp = ',lmfp,' m')

    #Brillouin avant
    if k is None:
        k = np.linspace(0,3*k0*dnsn**0.5,500)
    x=lmfp*Z[i]**0.5*k
    Ak =( 0.5 +isnonlocal*Z[i]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )

    #g0= np.sqrt(np.pi) *(cs/c)**3*(np.exp(-Am*1836*9.1e-31*cs**2/(2*Tim*1.6e-19))/(2*Tim/511000.)**1.5 +Am*1836./Zm/(2*Te/511000.)**1.5)
    vi=np.sqrt(Tim/(Am*1836*511000.))*c
    g0= np.sqrt(np.pi/8) *(cs/vi)**3*(np.exp(-0.5*(cs/vi)**2 ) +np.sqrt(Zm/(Am*1836.)) *(Zm*Te/Tim)**-1.5)

    print('g_0 = ', g0)

    if vpscs is None:
        vphi = np.linspace(0.*cs, 1.2*cs,100)
    else:
        vphi=vpscs*cs
    V, K=np.meshgrid(vphi,k)
    alphakold=alphak
    print('alpha[0] = ',alphak )
    alphak = alpha_kin(Te,Ti, Z, A, nisne, (vphi), k2lde2=0, ne=1)
    alphaf = 1./( 1-2*1j*g0*vphi/cs - (vphi/cs)**2  )
    print('alpha[0] = ',alphak[0] )
    print(np.shape(alphak), np.shape(Ak))
    #print('alphak = ',alphak)
    Up = 0*V +0j*V
    Ufp = 0*V +0j*V
    Um = 0*V +0j*V
    Ufm = 0*V +0j*V
    for iv in range(len(vphi)):
        for ik in range(len(k)): 
            #cinetique
            exp = np.exp( -1./(  nesnc*dnsn/n**2*alphak[iv]*Ak[ik] *w0**2 /(4*km*k[ik]*c**2 ) ) )
            E= 0.25 * ( (k[ik]/k0-1./f)**2 - exp*(k[ik]/k0+1./f )**2  ) /(1-exp)
            a=1
            b=-2*vphi[iv]/n/c
            C=(vphi[iv]/n/c)**2-E
            delta = b**2-4*a*C
            thd=np.arctan(np.imag(delta)/np.real(delta))+np.pi*(np.real(delta)<0)
            rd = np.abs(delta)
            Up[ik,iv] =  (-b+rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            Um[ik,iv] =  (-b-rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            
            #Fluide
            exp = np.exp( -1./( nesnc*dnsn/n**2*alphaf[iv]*Ak[ik] *w0**2 /(4*km*k[ik]*c**2 ) ) )
            E= 0.25 * ( (k[ik]/k0-1./f)**2 - exp*(k[ik]/k0+1./f )**2  ) /(1-exp)
            a=1
            b=-2*vphi[iv]/n/c
            C=(vphi[iv]/n/c)**2-E
            delta = b**2-4*a*C
            thd=np.arctan(np.imag(delta)/np.real(delta))+np.pi*(np.real(delta)<0)
            rd = np.abs(delta)
            Ufp[ik,iv] =  (-b+rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            Ufm[ik,iv] =  (-b-rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            
    U= Up*(np.imag(Up)>0) + Um*(np.imag(Um)>0)
    Uf= Ufp*(np.imag(Ufp)>0) + Ufm*(np.imag(Ufm)>0)
            
    gamma = K*np.imag(U)
    kpara = K*np.real(U)
    gammaf = K*np.imag(Uf)
    kparaf = K*np.real(Uf)
    #np.arctan2(np.real(U2),np.imag(U2))/2.
    #uabs = np.sqrt(np.abs(U2))
    #gamma = K*uabs*np.cos(uth)
    #kpara = -K*uabs*np.sin(uth)

    #mesh  = gamma>0
    #gamma *= mesh
    #kpara *= mesh
    for iv in range(len(vphi)):
        for ik in range(len(k)):
            if np.isnan(gamma[ik,iv]) or np.isnan(kpara[ik,iv]):
                gamma[ik,:iv] = 0
                kpara[ik,:iv] = 0

            if np.isnan(gammaf[ik,iv]) or np.isnan(kparaf[ik,iv]):
                gammaf[ik,:iv] = 0
                kparaf[ik,:iv] = 0


    #gmax=np.max([np.max(gamma),np.max(gammaf)])/k0
    #kmax=np.max([np.max(kpara),np.max(kparaf)])/k0
    #print('gamma/k0 = ',gamma /k0
    #print('kpara/k0 = ',kpara/k0
    if figure is not None:
        plt.rcParams.update({'font.size': 30})
        fig = plt.figure(figure,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        ivphi=np.argmin(np.abs(vphi))
        gf= gammaf[:,ivphi]/k0
        g = gamma[:,ivphi]/k0
        plt.plot(k/k0,gf, '--k',linewidth=2)
        plt.plot(k/k0,g, 'k',linewidth=2)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$\Gamma/k_0$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=gamma.T/k0
        #data = np.log10(gamma)
        if gmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =0, vmax =gmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if gticks is not None:
            plt.colorbar(cf0, ticks=gticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+2,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=-kpara.T/k0
        #data = np.log10(gamma)
        if kmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =-kmax, vmax =kmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if kticks is not None:
            plt.colorbar(cf0, ticks=kticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+3,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=gammaf.T/k0
        #data = np.log10(gamma)
        if gmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =0, vmax =gmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if gticks is not None:
            plt.colorbar(cf0, ticks=gticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+4,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=-kparaf.T/k0
        if kmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =-kmax, vmax =kmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if kticks is not None:
            plt.colorbar(cf0, ticks=kticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
        
        
        
    
def  filam_kinRPP_param(Te,ztesti=1.,Z=[1.],A=[1.],nesnc=0.1,I0=3e14,f=8.,k0=2*np.pi/0.35e-6,nisne=None,isnonlocal=True,k=None,figure=1):
    c=3e8
    km=k0/(2*f)
    nc=k0**2*c**2*8.85e-12*9.1e-31/(1.6e-19)**2
    print('nc = ',nc, ' m^-3')
    if nisne is None:
        nisne = [1./np.float(np.sum(np.array(Z)))]*np.array(Z)/np.array(Z)
        print('nisne = ',nisne)
    n=np.sqrt(1-nesnc)
    w0=k0*c/n
    if k is None:
        dnsnm = I0*1e4/(nc*c*np.min(Te)*1.6e-19)
        k = np.linspace(0,3*k0*dnsnm**0.5,300)
    
    
    
    vphi=0
    V, K=np.meshgrid(Te,k)
    Up = 0*V +0j*V
    Ufp = 0*V +0j*V
    Um = 0*V +0j*V
    Ufm = 0*V +0j*V
    # Boucle 
    for it in range(len(Te)):
        print('Te = ', Te[it],' eV, it = ',it,' /  ',len(Te))
        for ik in range(len(k)): 
            Ti = np.array(Z) * Te[it]/ztesti
            if np.size(Ti)==1:
                Ti=[Ti]
            dnsn = I0*1e4/(nc*c*Te[it]*1.6e-19)
            Tim, Am, Zm = np.mean(np.array(Ti)),np.mean(np.array(A)),np.mean(np.array(Z))
            cs = np.sqrt((Zm*Te[it]+3*Tim)/(1836.*511000.*Am ))*c
            m=0
            for i in range(len(Ti)):
                m+=Z[i]**2*nisne[i]*Te/Ti[i]
            alphak = m/(1.+m)
    
            ne=nesnc*nc/1e6
            #print([1.,A[0]*1836.],[-1.,Z[0]],[ne,ne/Z[0]],[Te[it],Ti[0]]
            r=taux_collisionnel(masse=[1.,A[0]*1836.],charge=[-1.,Z[0]],dens=[ne,ne*nisne[0]],temp=[Te[it],Ti[0]],vd=[0,0])
          
            lmfp=r['lmfp']*1e-2
            iak=0
            if len(Z)>1:
                for i in range(1,len(Z)):
                    r=taux_collisionnel(masse=[1.,A[i]*1836.],charge=[-1.,Z[i]],dens=[ne,ne*nisne[i]],temp=[Te[it],Ti[i]],vd=[0,0])
                    if lmfp  > r['lmfp']*1e-2:
                        lmfp = r['lmfp']*1e-2 
                        iak  = i
            #print('lmfp = ',r['lmfp'],' m,  iak = ',iak
            
            #Brillouin avant
            x=lmfp*Z[i]**0.5*k
            Ak =( 0.5 +isnonlocal*Z[i]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )
            g0= np.sqrt(np.pi) *(cs/c)**3*(np.exp(-Am*1836*9.1e-31*cs**2/(2*Tim*1.6e-19))/(2*Tim/511000.)**1.5 +Am*1836./Zm/(2*Te[it]/511000.)**1.5)
    
            #print('g_0 = ', g0

    
            alphakold=alphak
            alphak = alpha_kin(Te[it],Ti, Z, A, nisne, (vphi), k2lde2=0, ne=1)
            alphaf = 1./( 1-2*1j*g0*vphi/cs - (vphi/cs)**2  )

            #cinetique
            exp = np.exp( -1./(  nesnc*dnsn/n**2*alphak*Ak[ik] *w0**2 /(4*km*k[ik]*c**2 ) ) )
            E= 0.25 * ( (k[ik]/k0-1./f)**2 - exp*(k[ik]/k0+1./f )**2  ) /(1-exp)
            a=1
            b=-2*vphi/n/c
            C=(vphi/n/c)**2-E
            delta = b**2-4*a*C
            thd=np.arctan(np.imag(delta)/np.real(delta))+np.pi*(np.real(delta)<0)
            rd = np.abs(delta)
            Up[ik,it] =  (-b+rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            Um[ik,it] =  (-b-rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            
            #Fluide
            exp = np.exp( -1./( nesnc*dnsn/n**2*alphaf*Ak[ik] *w0**2 /(4*km*k[ik]*c**2 ) ) )
            E= 0.25 * ( (k[ik]/k0-1./f)**2 - exp*(k[ik]/k0+1./f )**2  ) /(1-exp)
            a=1
            b=-2*vphi/n/c
            C=(vphi/n/c)**2-E
            delta = b**2-4*a*C
            thd=np.arctan(np.imag(delta)/np.real(delta))+np.pi*(np.real(delta)<0)
            rd = np.abs(delta)
            Ufp[ik,it] =  (-b+rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            Ufm[ik,it] =  (-b-rd**0.5*np.exp(1j*thd/2.)) /(2*a)
            
    U= Up*(np.imag(Up)>0) + Um*(np.imag(Um)>0)
    Uf= Ufp*(np.imag(Ufp)>0) + Ufm*(np.imag(Ufm)>0)
            
    gamma = K*np.imag(U)
    kpara = K*np.real(U)
    gammaf = K*np.imag(Uf)
    kparaf = K*np.real(Uf)
    #np.arctan2(np.real(U2),np.imag(U2))/2.
    #uabs = np.sqrt(np.abs(U2))
    #gamma = K*uabs*np.cos(uth)
    #kpara = -K*uabs*np.sin(uth)

    
    #mesh  = gamma>0
    #gamma *= mesh
    #kpara *= mesh
    for it in range(len(Te)):
        for ik in range(len(k)):
            if np.isnan(gamma[ik,it]) or np.isnan(kpara[ik,it]):
                gamma[ik,:it] = 0
                kpara[ik,:it] = 0

            if np.isnan(gammaf[ik,it]) or np.isnan(kparaf[ik,it]):
                gammaf[ik,:it] = 0
                kparaf[ik,:it] = 0
                
    #Max
    kemax,kemaxf=0*Te,0*Te
    for it in range(len(Te)):
        ikm=np.argmax(np.abs(gamma[:,it]))
        kemax[it] = k[ikm]
        ikm=np.argmax(np.abs(gammaf[:,it]))
        kemaxf[it] = k[ikm]
    print(kemax/k0)
    #print('gamma/k0 = ',gamma /k0
    #print('kpara/k0 = ',kpara/k0
    if figure is None:
        return {'gamma':gamma.T,'gammaf':gammaf.T,'kemaxf':kemaxf,'kemax':kemax}
    else:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=gamma.T/k0
        #data = np.log10(gamma)
        cf0 = plt.pcolor(k/k0,Te,data,cmap=plt.cm.gist_earth_r)#, vmin =0, vmax =1.5*np.max(kfa*ufa/k0) )
        plt.colorbar(cf0, format='%.0e')#, ticks=ct)
        plt.plot(kemax/k0,Te,'k',linewidth=2)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$T_e$ (eV)")
        if Te[2]-Te[1]!=Te[1]-Te[0]:
            ax0.set_yscale('log')
        if k[2]-k[1]!=k[1]-k[0]:
            ax0.set_xscale('log')
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=gammaf.T/k0
        #data = np.log10(gamma)
        cf0 = plt.pcolor(k/k0,Te,data,cmap=plt.cm.gist_earth_r)#, vmin =0, vmax =1.5*np.max(kfa*ufa/k0) )
        plt.colorbar(cf0, format='%.0e')#, ticks=ct)
        plt.plot(kemaxf/k0,Te,'k',linewidth=2)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$T_e$ (eV)")
        if Te[2]-Te[1]!=Te[1]-Te[0]:
            ax0.set_yscale('log')
        if k[2]-k[1]!=k[1]-k[0]:
            ax0.set_xscale('log')
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
        
def nuIAW(Te,Ti,Z,A,k):
    c=3e8
    cs = np.sqrt((Z*Te+3*Ti)/(1836.*511000.*A ))*c
    g0= np.sqrt(np.pi) *(cs/c)**3*(np.exp(-A*1836*9.1e-31*cs**2/(2*Ti*1.6e-19))/(2*Ti/511000.)**1.5 +A*1836./Z/(2*Te/511000.)**1.5)
    return g0*cs*k

def xpFuchs(figure=1):
    k0=2*np.pi/1e-6
    Te,Ti = 140,140/10.
    ne=1.2e19
    nc=1e21
    Z,A=2.,4.
    k0*2*np.pi/1e-6
    I0=3.8e13
    f=1./90e-3
    
    te=np.linspace(60,140,100)
    k=np.logspace(2.5,6,400)
    r=filam_kinRPP_param(Z=[Z],A=[A],ztesti=Z*Te/Ti,Te=te,f=f, k0=k0,nesnc=ne/nc,I0=I0,k=k,figure=None)
    gamma=r['gammaf']
    kemax = r['kemaxf']
    
    r=filam_kinRPP_param(Z=[Z],A=[A],ztesti=Z*Te/Ti,Te=te, f=100,k0=k0,nesnc=ne/nc,I0=I0,k=k,figure=None)
    kemaxp = r['kemaxf']
    
    r=taux_collisionnel(masse=[1.,4*1836.],charge=[-1.,2],dens=[1e19,1e19/2.],temp=[10.,1.],vd=[0,0])
    nup = r['nup']
    t=np.linspace(-1e-9,1e-9,300)
    Tec=10+ (5./2.*I0*1e4/(1.6e-19)/3e8/(1.5*ne*1e6) * 10**1.5*0.5*ne/nc*nup * t )**(2./5.)
    tau = 500e-12*np.sqrt(np.log(2))
    Tec2=10+ (5./2.*I0*1e4/(1.6e-19)/3e8/(1.5*ne*1e6) * 10**1.5*0.5*ne/nc*nup *0.5*np.sqrt(np.pi)* tau*(erf(t/tau)-erf(t[0]/tau)) )**(2./5.)

    if figure is not None:
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=1., top=0.9, bottom=0.2)
        data=gamma*1e-3
        cf0 = plt.pcolor(2*np.pi/k*1e6,te,data,cmap=plt.cm.gist_earth_r)#, vmin =0, vmax =1.5*np.max(kfa*ufa/k0) )
        plt.colorbar(cf0)#, format='%.0e')#, ticks=ct)
        km ,=plt.plot(1e6*2*np.pi/kemax,te,'k',linewidth=2)
        kmp,=plt.plot(1e6*2*np.pi/kemaxp,te,'--k',linewidth=2)
        prop = fm.FontProperties(size=25)
        plt.legend([kmp,km],['$\mathrm{max}(\Gamma)$ \n plane wave' ,'$\mathrm{max}(\Gamma)$ \n RPP'],loc=9, prop=prop,frameon=False,bbox_to_anchor=(0.68, 1.05))
        ax0.set_xlabel("$\lambda_s$ ($\mu$m)")
        ax0.set_ylabel("$T_e$ (eV)")
        if te[2]-te[1]!=te[1]-te[0]:
            ax0.set_yscale('log')
        if k[2]-k[1]!=k[1]-k[0]:
            ax0.set_xscale('log')
        ax0.set_xlim(10,400)
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
    
        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        #kmp,=plt.plot(t*1e9-0.3,Tec,'k',linewidth=2)
        kmp,=plt.plot(t*1e9,Tec2,'k',linewidth=2)
        #prop = fm.FontProperties(size=25)
        #plt.legend([kmp,km],['$\mathrm{max}(\Gamma)$ \n plane wave' ,'$\mathrm{max}(\Gamma)$ \n RPP'],loc=9, prop=prop,frameon=False,bbox_to_anchor=(0.68, 1.05))
        ax0.set_xlabel("$t$ (ns)")
        ax0.set_ylabel("$T_e$ (eV)")
        #ax0.set_yscale('log')
        #ax0.set_xlim(10,400)
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()