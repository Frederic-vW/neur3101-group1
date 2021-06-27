#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Central Pattern Generator model
# FvW 06/2020

"""
Morris-Lecar model units
C: µF/cm^2
g: mS/cm^2
V: mV
I: µA/cm^2
"""

import os
#import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

fs_ = 20


def acov(x, lmax):
    """
    Autocovariance

    Parameters
    ----------
        x : 1D array of float
            input signal
        lmax : int
            maximum time lag

    Returns
    -------
        C : 1D array of float
            autocovariance coefficients
    """
    n = len(x)
    C = np.zeros(lmax)
    for k in range(lmax):
        C[k] = np.mean((x[0:n-k]-np.mean(x[0:n-k])) * (x[k:n]-np.mean(x[k:n])))
        #C[k] = np.mean(x[0:n-k]*x[k:n])
    # normalize
    C /= np.var(x)
    return C


def ccov(x, y, lmax):
    """
    Cross-covariance

    Parameters
    ----------
        x : 1D array of float
            input signal 1
        y : 1D array of float
            input signal 2
        lmax : int
            maximum time lag

    Returns
    -------
        C : 1D array of float
            autocovariance coefficients
    """
    #n = len(x)
    C = np.zeros(2*lmax+1)
    xm = (x-x.mean()) / x.std()
    ym = (y-y.mean()) / y.std()
    lags = np.array([k for k in range(-lmax,lmax+1)])
    for i, k in enumerate(lags):
        yk = np.roll(ym,k)
        C[i] = np.mean(xm * yk)
        #C[k] = np.mean((x[0:n-k]-np.mean(x[0:n-k])) * (x[k:n]-np.mean(x[k:n])))
        #C[k] = np.mean(x[0:n-k]*x[k:n])
    return lags, C


def check_exports():
    f = ("/home/frederic/Projects/UNSW/teaching/T2_2020/NEUR3101/2020/Pracs/"
         "P2_CPG/task1_data4.csv")
    x = np.loadtxt(f, skiprows=1)
    print(x.shape)
    plt.figure(figsize=(16,3))
    plt.plot(x[:,0], x[:,1], '-k', lw=2)
    plt.grid()
    plt.tight_layout()
    plt.show()


def locmax(x):
    """Get local maxima of 1D-array

    Args:
        x: numeric sequence
    Returns:
        m: list, 1D-indices of local maxima
    """
    dx = np.diff(x) # discrete 1st derivative
    zc = np.diff(np.sign(dx)) # zero-crossings of dx
    m = 1 + np.where(zc == -2)[0] # indices of local max.
    return m


def ml1(gL=2, VL=-60, gCa=4, VCa=120, gK=8, VK=-84, C=20, I_ext=None,
        V1=-1.2, V2=18, V3=2, V4=30, phi=0.04, sd=5, v0=-60, w0=0,
        T=100, dt=0.05, n0=0, doplot=False):
    """
    Single-cell Morris-Lecar dynamics

    Parameters
    ----------
    gL : float,  optional
        leak conductance
    VL : float,  optional
        reversal potential of leak current
    gCa : float,  optional
        Ca2+ conductance
    VCa : float,  optional
        reversal potential of Ca2+ current
    gK : float,  optional
        K+ conductance
    VK : float,  optional
        reversal potential of K+ current
    C : float,  optional
        membrane capacitance
    I_ext : float, list, tuple, nd.array, optional
        external current, if list, tuple or nd.array, use first value
    V1 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V2 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V3 : float, optional
        shape parameter for K+ channel steady-state open probability function
    V4 : float, optional
        shape parameter for K+ channel steady-state open probability function
    phi : float, optional
        parameter in the K+ current ODE
    sd : float, optional
        standard deviation of the noise source
    v0 : float, optional
        initial value of the voltage variable V
    w0 : float, optional
        initial value of the K+ current variable w
    T : float, optional
        total simulation time
    dt : float, optional
        sampling time interval
    v_mn : float, optional
        plotting limit: minimum voltage
    v_mx : float, optional
        plotting limit: maximum voltage
    w_mn : float, optional
        plotting limit: minimum K+ channel open fraction
    w_mx : float, optional
        plotting limit: maximum K+ channel open fraction
    doplot : boolean, optional
        construct the plot or not

    Returns
    -------
    X : 1D array of float
        voltage values
    Y : 1D array of float
        K+ channel open fraction

    Example
    -------
    >>> ...

    """
    nt = int(T/dt)
    #C1 = 1/C
    sd_sqrt_dt = sd*np.sqrt(dt)
    try:
        # in case I_ext is provided
        I = np.hstack( (I_ext[0]*np.ones(n0), I_ext) )
    except:
        # I_ext not provided, set to zero
        I = np.zeros(n0+nt)

    # initial conditions
    v = v0
    w = w0
    X = np.zeros(nt)
    X[0] = v
    Y = np.zeros(nt)
    Y[0] = w

    # steady-state functions
    m_inf = lambda v: 0.5*(1 + np.tanh((v-V1)/V2))
    w_inf = lambda v: 0.5*(1 + np.tanh((v-V3)/V4))
    lambda_w = lambda v: phi*np.cosh((v-V3)/(2*V4))

    for t in range(n0+nt):
        if (t%100 == 0): print(f"t={t:d}/{n0+nt:d}\r", end="")
        # Morris-Lecar equations
        dvdt = 1/C * (-gL*(v-VL) -gCa*m_inf(v)*(v-VCa) -gK*w*(v-VK) + I[t])
        dwdt = lambda_w(v) * (w_inf(v) - w)
        # integrate
        v += (dvdt*dt + sd_sqrt_dt*np.random.randn()) # Ito
        w += (dwdt*dt)
        if (t >= n0):
            X[t-n0] = v
            Y[t-n0] = w
    print("")

    if doplot:
        time = np.arange(nt)*dt
        fig, ax = plt.subplots(2,1,figsize=(16,8))
        # membrane voltage
        ax[0].plot(time, X, '-k', lw=2)
        ax[0].set_ylim(-100,60)
        ax[0].set_xlabel("time [ms]", fontsize=fs_)
        ax[0].set_ylabel("V [mV]", fontsize=fs_)
        #ax[0].set_title(f"Single cell Morris-Lecar", fontsize=fs_)
        ax[0].grid(True)
        # input current
        ax[1].plot(time, I[n0:], '-k', lw=2)
        #ax[1].set_ylim(-100,60)
        ax[1].set_xlabel("time [ms]", fontsize=fs_)
        ax[1].set_ylabel(r"$I_{ext} \; [\mu A/cm^2]$", fontsize=fs_)
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()

    return X, Y


def ml2(gL=2, VL=-60, gCa=4, VCa=120, gK=8, VK=-84, C=20, I_ext=None,
        V1=-1.2, V2=18, V3=2, V4=30, phi=0.04, sd=5, v0=-60, w0=0,
        T=100, dt=0.05, n0=0, coupling='mode-A', doplot=False):
    """
    Two cell Morris-Lecar model
    """
    nt = int(T/dt)
    sd_sqrt_dt = sd*np.sqrt(dt)
    try:
        # in case I_ext is provided
        I = np.hstack( (I_ext[0]*np.ones(n0), I_ext) )
    except:
        # I_ext not provided, set to zero
        I = np.zeros(n0+nt)

    if coupling == 'mode-A':
        J_01 = -0.05
        J_10 = -0.05
    elif coupling == 'mode-B':
        J_01 = 0.01
        J_10 = 0.01
    elif coupling == 'mode-C':
        J_01 = 0.01
        J_10 = -0.01
    else:
        J_01 = 0
        J_10 = 0

    # initial conditions
    v0, w0 = 0, 0 # neuron-1
    v1, w1 = 0, 0 # neuron-2
    X0 = np.zeros(nt)
    Y0 = np.zeros(nt)
    X1 = np.zeros(nt)
    Y1 = np.zeros(nt)

    # steady-state functions
    m_inf = lambda v: 0.5*(1 + np.tanh((v-V1)/V2))
    w_inf = lambda v: 0.5*(1 + np.tanh((v-V3)/V4))
    lambda_w = lambda v: phi*np.cosh((v-V3)/(2*V4))

    for t in range(n0+nt):
        if (t%100 == 0): print(f"t={t:d}/{n0+nt:d}\r", end="")
        # Morris-Lecar equations
        # neuron-1
        dv0dt = 1/C * (-gL*(v0-VL) -gCa*m_inf(v0)*(v0-VCa) -gK*w0*(v0-VK) \
                      + I[t]) + J_10*(v1-v0)
        dw0dt = lambda_w(v0) * (w_inf(v0) - w0)
        # neuron-2
        dv1dt = 1/C * (-gL*(v1-VL) -gCa*m_inf(v1)*(v1-VCa) -gK*w1*(v1-VK) \
                       + I[t]) + J_01*(v0-v1)
        dw1dt = lambda_w(v1) * (w_inf(v1) - w1)
        # integrate
        v0 += (dv0dt*dt + sd_sqrt_dt*np.random.randn()) # Ito
        w0 += (dw0dt*dt)
        v1 += (dv1dt*dt + sd_sqrt_dt*np.random.randn()) # Ito
        w1 += (dw1dt*dt)
        if (t >= n0):
            X0[t-n0] = v0
            X1[t-n0] = v1
            Y0[t-n0] = w0
            Y1[t-n0] = w1
    print("")

    if doplot:
        time = np.arange(nt)*dt
        fig, ax = plt.subplots(2, 1, figsize=(16,8))
        ax[0].plot(time, X0, '-k', lw=2)
        ax[0].grid(True)
        ax[0].set_xlabel("time [a.u.]", fontsize=fs_)
        ax[0].set_ylabel("voltage [a.u.]", fontsize=fs_)
        ax[0].set_title(f"Two cell Morris-Lecar", fontsize=fs_)
        ax[1].plot(time, X1, '-k', lw=2)
        ax[1].grid(True)
        ax[1].set_xlabel("time [a.u.]", fontsize=fs_)
        ax[1].set_ylabel("voltage [a.u.]", fontsize=fs_)
        plt.tight_layout()
        plt.show()

    return X0, Y0, X1, Y1


def ml8(gL=0.6, VL=-1.8, gCa=3, VCa=1, gK=1.8, VK=-0.8, C=1, I_ext=1.0,
        V1=0.2, V2=0.4, V3=0.3, V4=0.2, phi=0.2, sd=5, v0=-60, w0=0,
        T=100, dt=0.01, n0=0, mode=None, doplot=False, doanimate=False):
    """
    Central Pattern Generator for Quadruped locomotion
    based on the Morris-Lecar model

    References
    ----------

    .. [1] Buono L and Golubitsky M, "Models of central pattern generators for
           quadruped locomotion I. Primary Gaits". J. Math. Biol. 42,
           291--326 (2001)

    .. [2] Buono L and Golubitsky M, "Models of central pattern generators for
           quadruped locomotion II. Secondary Gaits". J. Math. Biol. 42,
           327--346 (2001)
    """

    """
    indices here: 0..7
    original paper: 1..8
    Returns:
        C: integer array (N,2)
           C[i,0]: neighbour `(i-2) mod N` of neuron `i`
           C[i,1]: neighbour `(i+(-1)^i) mod N` of neuron `i`

    explicitly:
    i | (i-2)%N | (i+eps_i)%N | (i+4)%N
    0      6           1          4
    1      7           0          5
    2      0           3          6
    3      1           2          7
    4      2           5          0
    5      3           4          1
    6      4           7          2
    7      5           6          3
    """
    if (mode == 'A'):
        mode_ = 'walk'
    elif (mode == 'B'):
        mode_ = 'jump'
    elif (mode == 'C'):
        mode_ = 'pace'
    elif (mode == 'D'):
        mode_ = 'bound'
    elif (mode == 'E'):
        mode_ = 'trot'
    elif (mode == 'F'):
        mode_ = 'pronk'
    else:
        print("ERROR: select one of the modes: 'A', 'B', 'C', 'D', 'E', 'F'")
        sys.exit()

    N = 8
    K = np.zeros((8,2),dtype=np.int)
    for i in range(N):
        k = (i-2)%N
        ei = (-1)**i
        l = (i + ei)%N
        #m = (i + 4)%N
        #print(f"i = {i:d}, {k:d} > {i:d}, {l:d} > {i:d}")
        #print(f"i = {i:d}, {k:d}, {l:d}, {m:d}")
        K[i,0] = k
        K[i,1] = l

    # [alpha, beta, gamma, delta]
    modes = {}
    modes['pronk'] = [0.2, 0.2, 0.2, 0.2]
    modes['pace'] = [0.2, 0.2, -0.2, -0.2]
    modes['bound'] = [-0.2, -0.2, 0.2, 0.2]
    modes['trot'] = [-0.6, -0.6, -0.6, -0.6]
    modes['jump'] = [0.01, -0.01, 0.2, 0.2]
    modes['walk'] = [0.01, -0.01, -1.2, -1.2]
    modes['canter'] = [0.17, -0.2, -0.9, -1]
    if mode_ == 'canter':
        gCa = 8
    modes['runwalk'] = [-0.78, -0.56, 0.12, -1.14]
    if mode_ == 'runwalk':
        gCa = 2
    modes['doublebond'] = [-0.6, -0.77, 0.3, 0.5]
    if mode_ == 'doublebond':
        gCa = 3

    # example initial conditions
    ic = {}
    ic["pronk"] = {}
    ic["pronk"]["v"] = [-0.51, 1.84, 0.17, 0.35, -1.86, 1.87, -1.03, 0.65]
    ic["pronk"]["w"] = [0.25, 1.13, 0.97, -0.41, -1.88, -1.18, -0.05, 1.43]

    # pace
    ic["pace"] = {}
    ic["pace"]["v"] = [-0.15, 0.11, -1.79, 0.84, 1.26, 0.16, -0.40, -1.65]
    ic["pace"]["w"] = [-0.90, -0.32, 0.772, 0.12, -0.31, 0.99, -1.79, -0.10]

    # bound
    ic["bound"] = {}
    ic["bound"]["v"] = [0.42, -0.40, -0.24, -0.51, 1.40, 2.40, -0.50, 0.87]
    ic["bound"]["w"] = [-0.74, -0.56, -1.29, 1.18, 0.27, 1.91, 0.80, -0.30]

    # trot
    ic["trot"] = {}
    ic["trot"]["v"] = [1.18, 0.91, 0.43, -1.61, -0.43, 1.34, -0.59, -0.68]
    ic["trot"]["w"] = [-0.26, -0.41, 0.49, -2.12, 0.36, 1.50, 1.06, -1.09]

    # jump
    ic["jump"] = {}
    ic["jump"]["v"] = [0.00, 0.37, 0.19, -1.46, -0.33, -1.00, -2.23, -1.20]
    ic["jump"]["w"] = [0.03, -0.90, -0.27, 0.81, -0.51, 0.30, -0.12, -0.02]

    # walk
    ic["walk"] = {}
    ic["walk"]["v"] = [-1.97, 0.31, 0.74, 0.43, -0.19, -1.42, 1.13, 1.32]
    ic["walk"]["w"] = [-0.24, -0.44, -0.52, -0.51, -0.54, -0.19, 0.26, -0.79]

    # canter
    ic["canter"] = {}
    ic["canter"]["v"] = [0.4, 0, 0, 0, 0, 0, 0, 0]
    ic["canter"]["w"] = [0.3, 0, 0, 0, 0, 0, 0, 0]

    # runwalk
    ic["runwalk"] = {}
    ic["runwalk"]["v"] = [-1.2147809, 0, 0, 0, 0, 0, 0, 0]
    ic["runwalk"]["w"] = [-0.058746844, 0, 0, 0, 0, 0, 0, 0]

    # doublebond
    ic["doublebond"] = {}
    ic["doublebond"]["v"] = [-1.0, 0, 0, 0, 0, 0, 0, 0]
    ic["doublebond"]["w"] = [0.6, 0, 0, 0, 0, 0, 0, 0]

    # apply inital conditions
    v = np.array(ic[mode_]["v"])
    w = np.array(ic[mode_]["w"])

    if mode_ not in modes:
        print("ERROR: mode not defined")
        sys.exit()

    #print(f"[+] mode: {mode_:s}")
    alpha, beta, gamma, delta = modes[mode_]

    n_t = int(T/dt)
    X = np.zeros((n_t,N))
    dvdt = np.zeros(N)
    dwdt = np.zeros(N)

    # steady-state functions
    m_inf = lambda v: 0.5*( 1 + np.tanh((v-V1)/V2) )
    w_inf = lambda v: 0.5*( 1 + np.tanh((v-V3)/V4) )
    lambda_w = lambda v: phi*np.cosh((v-V3)/(2*V4))

    for t in range(n0+n_t):
        # single node dynamics
        dvdt = -gCa*m_inf(v)*(v-VCa) -gL*(v-VL) -gK*w*(v-VK) + I_ext
        dwdt = lambda_w(v) * (w_inf(v) - w)
        # changes due to connections
        dvdt += (alpha*v[K[:,0]] + gamma*v[K[:,1]]) #+xi*x(i+4)
        dwdt += (beta*w[K[:,0]] + delta*w[K[:,1]]) #+eta*y(i+4)
        # integrate
        v += (dvdt*dt)
        w += (dwdt*dt)
        if (t >= n0):
            X[t-n0,:] = v
            #Y[t-n0,:] = w

    # interpolate
    downsample_fct = 5
    dt_interp = downsample_fct*dt
    n_interp = int(n_t/downsample_fct)
    t_ = np.arange(n_t)
    f_ip = interp1d(t_, X, axis=0, kind='linear')
    t_new = np.linspace(0, n_t-1, n_interp)
    X = f_ip(t_new)
    data = {}
    data['LH'] = X[:,0] # left hind
    data['RH'] = X[:,1] # right hind
    data['LF'] = X[:,2] # left fore
    data['RF'] = X[:,3] # right fore
    data['time'] = dt_interp*np.arange(n_interp)

    #doanimate0 = doanimate1 = doanimate2 = doanimate3 = False

    if doplot:
        # 0: left hind leg
        # 1: right hind leg
        # 2: left fore leg
        # 3: right fore leg
        p_ann = {
            'xy' : (0.01,0.80),
            'xycoords' : 'axes fraction',
            'fontsize' : 22,
            'fontweight' : 'bold'
        }
        fig, ax = plt.subplots(4, 1, figsize=(16,8), sharex=True)
        legs = ['LF', 'RF', 'LH', 'RH']
        for i, leg in enumerate(legs):
            ax[i].plot(data[leg], '-k')
            ax[i].annotate(leg, **p_ann)
            ax[i].grid()
        #ax[0].set_title(f"{mode_:s}", fontsize=22)
        plt.tight_layout()
        plt.show()

    if doanimate:
        """
        make movie
        """
        #movname = f"cpg_quad_{mode_:s}.mp4"
        movname = f"cpg_mode-{mode:s}.mp4"
        print(f"[+] Animate data as movie: {movname:s}")
        loc = {
            "LF" : (20,40),
            "RF" : (150,40),
            "LH" : (20,125),
            "RH" : (155,125),
        }
        # find local maxima in activation time courses
        scf = 10
        peaks = {}
        for leg in data:
            pks = locmax(data[leg])
            pks = pks/scf
            pks = pks.astype('int')
            peaks[leg] = pks
        # load animal background image
        #img = mpimg.imread('./img/quadruped.png')
        #img = mpimg.imread('./img/quadruped_bw.png')
        img = mpimg.imread('./img/gecko2.png')

        #fig, ax = plt.subplots()
        fig = plt.figure(figsize=(6,6))
        ax = plt.gca()
        ax.imshow(img)
        ax.axis('off')
        s = 600
        lf = ax.scatter(loc['LF'][0], loc['LF'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')
        rf = ax.scatter(loc['RF'][0], loc['RF'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')
        lh = ax.scatter(loc['LH'][0], loc['LH'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')
        rh = ax.scatter(loc['RH'][0], loc['RH'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')

        def update(i):
            if (i%10 == 0): print(f"\tt = {i:d}/{n2:d}\r", end="")
            if i in peaks['LF']:
                lf.set_facecolors('r')
            else:
                lf.set_facecolors('none')
            if i in peaks['RF']:
                rf.set_facecolors('r')
            else:
                rf.set_facecolors('none')
            if i in peaks['LH']:
                lh.set_facecolors('r')
            else:
                lh.set_facecolors('none')
            if i in peaks['RH']:
                rh.set_facecolors('r')
            else:
                rh.set_facecolors('none')

        # make animation
        n_interp = len(data['LF'])
        n2 = n_interp//scf
        ani = FuncAnimation(fig, update, interval=50, save_count=n2)
        ani.save(movname, dpi=300)
        plt.show()
        print("Animation created and saved.")

    return data


def mod(x,N):
    """
    modulo function
    """
    if (x <= 0): x += N
    if (x > N): x -= N
    return x


def nonlinearity():
    """
    Study the non-linear membrane behaviour in the ML membrane model
    """
    dt = 0.05
    T = 500
    t_on, t_off = 50, 100
    #t_on, t_off = 50, 100
    i_on, i_off = int(t_on/dt), int(t_off/dt)
    nI = 50 # 50
    Is = np.linspace(10,150,nI) # input current range
    vmax = np.zeros(nI) # max. voltage
    for i, i_ext in enumerate(Is):
        print(f"i = {i:d}/{nI:d}\r", end="")
        I_ext = np.zeros(int(T/dt))
        I_ext[i_on:i_off] = i_ext
        params = {'T': T, 'dt': dt, 'sd': 0.05, 'I_ext': I_ext, 'doplot': False}
        X, Y = ml1(**params)
        vmax[i] = X.max()
    print("\n")

    plt.figure(figsize=(6,6))
    plt.plot(Is, vmax, '-ok')
    plt.grid()
    plt.tight_layout()
    plt.show()


def phase_plane(v=np.array([]), w=np.array([]),
                gL=2, VL=-60, gCa=4, VCa=120, gK=8, VK=-84, C=20, I_ext=0,
                V1=-1.2, V2=18, V3=2, V4=30, phi=0.04, sd=5, v0=-60, w0=0,
                T=100, dt=0.05, n0=None,
                v_mn=None, v_mx=None, w_mn=None, w_mx=None,
                doplot=True):
    """
    Make a (V,w) phase-plane plot

    Plot the vector field defined by the Morris-Lecar equations for
    dV/dt and dw/dt (streamlines), the V-nullcline, the w-nullcline, and
    the trajectory defined by the data points.

    Parameters
    ----------
    v : 1D array, optional
        time course of voltage values; if empty, not trajectory is plotted
    w : 1D array, optional
        time course of open K+ fraction; if empty, not trajectory is plotted
    gL : float,  optional
        leak conductance
    VL : float,  optional
        reversal potential of leak current
    gCa : float,  optional
        Ca2+ conductance
    VCa : float,  optional
        reversal potential of Ca2+ current
    gK : float,  optional
        K+ conductance
    VK : float,  optional
        reversal potential of K+ current
    C : float,  optional
        membrane capacitance
    I_ext : float, list, tuple, nd.array, optional
        external current, if list, tuple or nd.array, use first value
    V1 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V2 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V3 : float, optional
        shape parameter for K+ channel steady-state open probability function
    V4 : float, optional
        shape parameter for K+ channel steady-state open probability function
    phi : float, optional
        parameter in the K+ current ODE
    sd : float, optional
        standard deviation of the noise source
    v0 : float, optional
        initial value of the voltage variable V
    w0 : float, optional
        initial value of the K+ current variable w
    T : float, optional
        total simulation time
    dt : float, optional
        sampling time interval
    v_mn : float, optional
        plotting limit: minimum voltage
    v_mx : float, optional
        plotting limit: maximum voltage
    w_mn : float, optional
        plotting limit: minimum K+ channel open fraction
    w_mx : float, optional
        plotting limit: maximum K+ channel open fraction
    doplot : boolean, optional
        construct the plot or not

    Example
    -------
    >>> ...

    """
    # plot phase-plane for first current value if I_ext is an array
    if isinstance(I_ext, (list, tuple, np.ndarray)):
        I_ext = I_ext[0]
    C_ = 1./C
    #print(f"u_mn = {u_mn:.3f}, u_mx = {u_mx:.3f}")
    #print(f"v_mn = {v_mn:.3f}, v_mx = {v_mx:.3f}")
    if v_mn == None:
        v_mn = v.min()
    if v_mx == None:
        v_mx = v.max()
    if w_mn == None:
        w_mn = w.min()
    if w_mx == None:
        w_mx = w.max()
    v_ = np.arange(v_mn, v_mx, 0.05)
    w_ = np.arange(w_mn, w_mx, 0.05)
    v_null_x = w_null_x = v_
    V, W = np.meshgrid(v_, w_)

    # background color
    #c_int = 128 # gray intensity as integer
    #c_hex = '#{:s}'.format(3*hex(c_int)[2:])
    cmap = matplotlib.cm.get_cmap('gnuplot2')

    v_null_y = (-gL*(v_-VL) -gCa*0.5*(1+np.tanh((v_-V1)/V2))*(v_-VCa) \
                + I_ext) / (gK*(v_-VK))
    w_null_y = 0.5*(1+np.tanh((v_-V3)/V4))
    Dv = C_ * (-gL*(V-VL) -gCa*0.5*(1+np.tanh((V-V1)/V2))*(V-VCa) \
               -gK*W*(V-VK) + I_ext)
    Dw = phi*np.cosh((V-V3)/(2*V4))*(0.5*(1+np.tanh((V-V3)/V4))-W)
    #V, W = np.meshgrid(v_, w_)

    # +++ Figure +++
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    #ax.patch.set_facecolor(c_hex)
    plt.plot(v_, np.zeros_like(v_), '-k', lw=1) # x-axis
    plt.plot(np.zeros_like(w_), w_, '-k', lw=1) # y-axis
    # nullcline: dV/dt = 0
    plt.plot(v_null_x, v_null_y, '-k', lw=3, label=r"$\frac{dV}{dt}=0$")
    # nullcline: dw/dt = 0
    plt.plot(w_null_x, w_null_y, '--k', lw=3, label=r"$\frac{dw}{dt}=0$")
    ps = {'x': v_, 'y': w_, 'u': Dv, 'v': Dw, 'density': 1.0, \
          'color': 'k', 'linewidth': 0.3, 'arrowsize': 0.80}
    p_ = plt.streamplot(**ps)
    # color nodes according to index j
    v_norm = (v-v_mn)/(v_mx-v_mn)
    w_norm = (w-w_mn)/(w_mx-w_mn)
    plt.scatter(v[::10], w[::10], marker='o', c='b', s=30, alpha=0.4)
    txt = r"$(v_0,w_0)$"
    #txt = r"$(v_0,w_0)=$" + f"({v0:.2f},{w0:.2f})"
    plt.annotate(txt, xy=(v0,w0), xycoords="data", fontsize=16, color="blue")

    #plt.grid(True)
    plt.axis([v_mn, v_mx, w_mn, w_mx])
    plt.xlabel("V", fontsize=18)
    plt.ylabel("w", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(loc=1, fontsize=18)
    #plt.axis('equal')
    plt.tight_layout()
    plt.show()


def population_code():
    """
    Construct a response plot for set of neurons in different grasp directions
    """
    N = 8
    print("Number of neurons: ", N)
    phi = 2*np.pi*np.arange(N)/N
    print("Principal directions: ", phi/(2*np.pi))

    # test directions
    M = 60
    psi = 2*np.pi*np.arange(M)/M

    # response array
    resp = np.zeros((N,M))

    q0 = 1.5 # basal firing rate
    T = 50 # simulation time
    for i, phi_ in enumerate(phi):
        for j, psi_ in enumerate(psi):
            # Poisson point process with angle-dependent rate
            q = q0 + np.cos(phi_ - psi_) # scalar product -> cos(phi-psi)
            # exponentially distributed waiting times
            # list of lists, firing times for each neuron
            t = []
            while (np.sum(t) < T):
                r = np.random.rand() # uniform random variable ~ U[0,1]
                t.append(-np.log(r)/q) # add exp. distributed waiting time
            resp[i,j] = len(t)/T

    rmax = resp.max()
    resp = 0.6*resp/rmax

    # make response plot
    d = 0.03

    #''' linear response plot
    plt.figure(figsize=(16,3))
    for i, phi_ in enumerate(phi):
        r_mean_x = 0
        r_mean_y = 0
        x, y = i, 1
        plt.plot(x, y, 'ok', ms=10)
        for j, psi_ in enumerate(psi):
            resp_ = resp[i,j]
            dx = resp_*np.cos(psi_)
            dy = resp_*np.sin(psi_)
            r_mean_x += rmax*dx
            r_mean_y += rmax*dy
            plt.plot([x, x+dx], [y, y+dy], '-k', lw=1, alpha=0.5)
        r_mean_x /= M
        r_mean_y /= M
        plt.plot([x, x+r_mean_x], [y, y+r_mean_y], '-b', lw=4, alpha=1)
        #plt.annotate(f"N{i+1:d}", xy=(x+d,y+d), xycoords="data", c="blue", fontsize=18)
        plt.annotate(f"N{i+1:d}", xy=(x,0.5), xycoords="data", c="blue", fontsize=18)
    #plt.axis('off')
    ax = plt.gca()
    ax.set_xticklabels([])
    ymn, ymx = ax.get_ylim()
    plt.ylim(0.4, ymx)
    plt.tight_layout()
    #plt.show()
    #'''

    #''' circular response plot
    plt.figure(figsize=(8,8))
    for i, phi_ in enumerate(phi):
        r_mean_x = 0
        r_mean_y = 0
        x, y = np.cos(phi_), np.sin(phi_)
        plt.plot(x, y, 'ok', ms=10)
        for j, psi_ in enumerate(psi):
            resp_ = resp[i,j]
            dx = resp_*np.cos(psi_)
            dy = resp_*np.sin(psi_)
            r_mean_x += rmax*dx
            r_mean_y += rmax*dy
            plt.plot([x, x+dx], [y, y+dy], '-k', lw=1, alpha=0.5)
        r_mean_x /= M
        r_mean_y /= M
        plt.plot([x, x+r_mean_x], [y, y+r_mean_y], '-b', lw=4, alpha=1)
        #plt.annotate(f"N{i+1:d}", xy=(x+d,y+d), xycoords="data", c="blue", \
        #             fontsize=18)
        plt.annotate(f"N{i+1:d}", xy=(0.6*x,0.6*y), xycoords="data", c="blue", \
                     fontsize=18)
        #plt.annotate(f"N{i+1:d}", xy=(x+r_mean_x*np.cos(phi_),y+r_mean_y*np.sin(phi_)), \
        #             xycoords="data", c="blue", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    #'''


def raster_plot():
    """
    Construct a raster plot for a given grasp direction
    """
    N = 8
    #print("Number of neurons: ", N)
    phi = 2*np.pi*np.arange(N)/N
    #print("Principal directions: ", phi/(2*np.pi))

    M = 6
    psi = 2*np.pi*np.arange(M)/M
    #print("Grasp angles: ", psi/(2*np.pi))
    phi0 = psi[1]
    #print(f"phi0: {phi0}")

    # Poisson point process with angle-dependent rate
    q0 = 1.5 # basal firing rate
    q = q0 + np.cos(phi-phi0) # scalar product -> cos(phi-psi)
    #print("\nPoisson rates:")
    #for i in range(N): print(f"q({i:d}) = {q[i]:.2f}")

    # exponentially distributed waiting times
    T = 50 # simulation time
    t = [[] for i in range(N)] # list of lists, firing times for each neuron
    for i in range(N):
        while (np.sum(t[i]) < T):
            r = np.random.rand() # uniform random variable ~ U[0,1]
            t[i].append(-np.log(r)/q[i]) # add exp. distrib. waiting time
        # waiting times (ISI) => firing times, max. time < T
        t[i] = np.cumsum(t[i][:-1]).tolist()
        #print(t[i])
        #print(np.max(t[i]))

    # make raster plot
    scf = 10
    plt.figure(figsize=(16,8))
    for i in range(N):
        y = 0.5*i
        for t_ in t[i]:
            plt.plot([t_/scf, t_/scf], [y, y+0.1], '-k') # vertical sticks
    ax = plt.gca()
    ax.set_yticks(np.arange(N)*0.5)
    ax.set_yticklabels([f"Neuron-{N-i:d}" for i in range(N)], fontsize=14)
    plt.xlabel("time [s]")
    plt.title("Raster plot")
    plt.show()

    for i in range(N):
        ri = scf*len(t[i]) / T
        print(f"Neuron #{i:d} average spike rate (1/s): {ri:.2f}")


def zcross(x, mode):
    """Get zero crossings of a vector
    mode: 'np': '-' => '+'
          'pn': '+' => '-'
          'all': all
    FvW 08-2007
    """
    zc = np.diff(np.sign(x))
    if ( mode == "pn" ):
        zc = 1+np.where(zc == -2)[0]
    elif ( mode == "np" ):
        zc = 1+np.where(zc == 2)[0]
    elif ( mode == "all" ):
        zc = 1+np.where(np.abs(zc) == 2)[0]
    else:
        zc = ([], )
    #zc = np.array(zc)
    return zc


def wrap_up():
    """
    Make figures for wrap-up session
    """
    os.chdir(("/home/frederic/Projects/UNSW/teaching/T2_2020/NEUR3101/2020/"
              "Pracs/P2_CPG/data/"))

    #''' Task-1
    f = "task1_data1.csv"
    #f = "task1_data2.csv"
    #f = "task1_data3.csv"
    #f = "task1_data4.csv"
    x = np.loadtxt(f, skiprows=1)
    plt.figure(figsize=(16,3))
    plt.plot(x[:,0], x[:,1], '-k', lw=2)
    plt.xlabel("time [a.u.]", fontsize=fs_)
    plt.ylabel("voltage [a.u.]", fontsize=fs_)
    plt.annotate(f, xy=(0.8,0.8,), xycoords='axes fraction', fontsize=fs_)
    plt.grid()
    plt.tight_layout()
    plt.show()
    #'''

    ''' Task-2
    f = "task2_data1.csv"
    #f = "task2_data2.csv"
    x = np.loadtxt(f, skiprows=1)
    print(x.shape)
    fig, ax = plt.subplots(2,1,figsize=(16,6))
    ax[0].plot(x[:,0], x[:,1], '-k', lw=2)
    ax[1].plot(x[:,0], x[:,2], '-k', lw=2)
    ax[1].set_xlabel("time [a.u.]", fontsize=fs_)
    ax[0].set_ylabel("voltage [a.u.]", fontsize=fs_)
    ax[1].set_ylabel("voltage [a.u.]", fontsize=fs_)
    #plt.annotate(f, xy=(0.8,0.8,), xycoords='axes fraction', fontsize=fs_)
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.show()
    '''

    ''' Task-3
    #f = "task3_data_mode-A.csv" # Pronk
    #f = "task3_data_mode-B.csv" # Pace
    #f = "task3_data_mode-C.csv" # Bound
    #f = "task3_data_mode-D.csv" # Trot
    f = "task3_data_mode-E.csv" # Jump
    #f = "task3_data_mode-F.csv" # Walk
    X = np.loadtxt(f, skiprows=1)
    print(X.shape)

    fig, ax = plt.subplots(4,1,figsize=(16,8), sharex=True)
    ax[0].plot(X[:,0], X[:,1], '-k', lw=2, label="left front")
    ax[1].plot(X[:,0], X[:,2], '-k', lw=2, label="right front")
    ax[2].plot(X[:,0], X[:,3], '-k', lw=2, label="left rear")
    ax[3].plot(X[:,0], X[:,4], '-k', lw=2, label="right rear")
    ax[3].set_xlabel("time [a.u.]", fontsize=fs_)
    for i in range(4):
        ax[i].grid(True)
        ax[i].legend(loc=2, fontsize=fs_+4)
    plt.tight_layout()
    plt.show()
    '''


def main():
    pass
    # 'pronk', 'pace', 'bound', 'trot', 'jump', 'walk'
    #q = Quad(mode='walk', T=500, dt=0.05)
    #check_exports()
    #wrap_up()
    #ml1(T=1000, dt=0.05, sd=0.1, I0=90, I1=110)
    #ml2(T=1000, dt=0.05, sd=0.1, I0=90, I1=120, coupling='mode-B')
    #data = ml8(n0=3000, mode="walk", doplot=True, doanimate=True)
    #data = ml8(T=200, dt=0.01, n0=3000, mode='E', doplot=True, doanimate=True)
    '''
    A: walk
    B: jump
    C: pace
    D: bound
    E: trot
    F: pronk
    '''
    #bifurcation_diagram()
    #oscillator_types(mode="type2")
    #anodal_break()
    #nonlinearity()
    population_code()

    '''
    task = 'Task3--'
    save_ = not True

    T, dt = 100, 0.05
    sd = 0.1

    if task == 'Task1':
        #I0 = I1 = -0.1; id=2
        #I0 = I1 = 0.01; id=1
        I0 = I1 = 0.25; id=4
        #I0 = I1 = 1.00; id=3
        X, Y = fhn1(T, dt, sd, I0, I1, doplot=True)
        plt.xcorr(X,X,maxlags=500); plt.show() # cross-correlation
        time = np.arange(X.shape[0])*dt
        fname = f"task1_data{id:d}.csv"
        if save_:
            print(f"[+] save data as: {fname:s}")
            np.savetxt(fname, np.vstack((time,X)).T, fmt='%.5f', delimiter=' ',\
                       newline='\n', header='time, voltage', footer='', \
                       comments='# ', encoding=None)

    if task == 'Task2':
        #I0 = I1 = -2.8
        #X0, Y0, X1, Y1 = fhn2(T, dt, sd, I0, I1, coupling='mode-A', doplot=True)
        I0 = I1 = 0.05
        X0, Y0, X1, Y1 = fhn2(T, dt, sd, I0, I1, coupling='mode-B', doplot=True)
        time = np.arange(X0.shape[0])*dt
        fname = "task2_data2.csv"
        if save_:
            np.savetxt(fname, np.vstack((time,X0,X1)).T, fmt='%.5f', \
                       delimiter=' ', newline='\n', \
                       header='time, V1, V2', footer='', comments='# ', \
                       encoding=None)

    if task == 'Task3':
        X = fhn8(mode='F', doplot=True)
        time = np.arange(X.shape[0])*dt
        fname = "task3_data_mode-F.csv"
        if save_:
            np.savetxt(fname, np.vstack((time,X[:,:4].T)).T, fmt='%.5f', \
                       delimiter=' ', newline='\n', \
                       header='time, V1, V2, V3, V4', footer='', comments='# ',\
                       encoding=None)
    '''


if __name__ == "__main__":
    os.system("clear")
    fs_ = 20
    main()
