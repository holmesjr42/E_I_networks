# for fast array manipulation
import numpy as np
# for plotting
import matplotlib.pyplot as plt
# for numerical ODE integration
from scipy.integrate import odeint
# for nonlinear equations
from scipy.optimize import fsolve
# to display plots in-line
# %matplotlib inline
# widget
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames

# parameters
tau_e = 20
tau_i = 10

# stimulus orientation
phi_1=45
phi_2=135  # if you need two stimuli
j_ee = 0.044
j_ie = 0.042
j_ei = 0.023
j_ii = 0.018
sigm_ori=32
sigm_ff=30
k=0.04
n=2.0
N=180


# array preparaton

# time parameters
t0 = 0
tmax = 100
tstep = 0.1
t = np.arange(t0, tmax, tstep)
c = np.arange(1, 101, 1)

# initial values
# stimulus 45

r_e_1=np.zeros((N,t.shape[0]))
r_i_1=np.zeros((N,t.shape[0]))
in_e_1=np.zeros((N,t.shape[0]))
in_i_1=np.zeros((N,t.shape[0]))
w_ee=np.zeros((N,N))
w_ei=np.zeros((N,N))
w_ie=np.zeros((N,N))
w_ii=np.zeros((N,N))
r_e_ss_1=np.zeros((N,t.shape[0]))
r_i_ss_1=np.zeros((N,t.shape[0]))
ext_in_1=np.zeros((N,100))
R_e_1=np.zeros((N,c.shape[0]))
R_i_1=np.zeros((N,c.shape[0]))
R_e_n_e_1=np.zeros((N,c.shape[0]))
R_e_n_i_1=np.zeros((N,c.shape[0]))
R_i_n_e_1=np.zeros((N,c.shape[0]))
R_i_n_i_1=np.zeros((N,c.shape[0]))
R_e_t_1=np.zeros((N,c.shape[0]))
R_i_t_1=np.zeros((N,c.shape[0]))

# stimulus 135
r_e_2=np.zeros((N,t.shape[0]))
r_i_2=np.zeros((N,t.shape[0]))
in_e_2=np.zeros((N,t.shape[0]))
in_i_2=np.zeros((N,t.shape[0]))
r_e_ss_2=np.zeros((N,t.shape[0]))
r_i_ss_2=np.zeros((N,t.shape[0]))
ext_in_2=np.zeros((N,100))
R_e_2=np.zeros((N,c.shape[0]))
R_i_2=np.zeros((N,c.shape[0]))
R_e_n_e_2=np.zeros((N,c.shape[0]))
R_e_n_i_2=np.zeros((N,c.shape[0]))
R_i_n_e_2=np.zeros((N,c.shape[0]))
R_i_n_i_2=np.zeros((N,c.shape[0]))
R_e_t_2=np.zeros((N,c.shape[0]))
R_i_t_2=np.zeros((N,c.shape[0]))

# stimulus 45+135
r_e_3=np.zeros((N,t.shape[0]))
r_i_3=np.zeros((N,t.shape[0]))
in_e_3=np.zeros((N,t.shape[0]))
in_i_3=np.zeros((N,t.shape[0]))
r_e_ss_3=np.zeros((N,t.shape[0]))
r_i_ss_3=np.zeros((N,t.shape[0]))
ext_in_3=np.zeros((N,100))
R_e_3=np.zeros((N,c.shape[0]))
R_i_3=np.zeros((N,c.shape[0]))
R_e_n_e_3=np.zeros((N,c.shape[0]))
R_e_n_i_3=np.zeros((N,c.shape[0]))
R_i_n_e_3=np.zeros((N,c.shape[0]))
R_i_n_i_3=np.zeros((N,c.shape[0]))
R_e_t_3=np.zeros((N,c.shape[0]))
R_i_t_3=np.zeros((N,c.shape[0]))

# firing rate sum
sum_ext=np.zeros(c.shape[0])
sum_net_e_e=np.zeros(c.shape[0])
sum_net_e_i=np.zeros(c.shape[0])
sum_net_i_e=np.zeros(c.shape[0])
sum_net_i_i=np.zeros(c.shape[0])

# function for the shortest distance

def sh_d(x):
    x=np.where(x<=90,x,(180-x))
    return x
#going to calculate the external input first
h_b=np.arange(1,181,1)
h_d1a=np.absolute(np.subtract(h_b,phi_1))
h_d1=sh_d(h_d1a)
h_d2a=np.absolute(np.subtract(h_b,phi_2))
h_d2=sh_d(h_d2a)
h_1=np.exp(-((h_d1**2)/(2*(sigm_ff**2))))
h_2=np.exp(-((h_d2**2)/(2*(sigm_ff**2))))
#equations
#weights
w_1=np.arange(1,181,1)
w_c,w_r=np.meshgrid(w_1,w_1)
w_da=np.absolute(w_r-w_c)
w_d=sh_d(w_da)
w_ee=j_ee*np.exp(-((w_d**2)/(2*(sigm_ori**2))))
w_ie=j_ie*np.exp(-((w_d**2)/(2*(sigm_ori**2))))
w_ei=j_ei*np.exp(-((w_d**2)/(2*(sigm_ori**2))))
w_ii=j_ii*np.exp(-((w_d**2)/(2*(sigm_ori**2))))
#euler
#this iteration may detect only the steady state value
dt=t.shape[0]-1
#make function for the stimulus orientation?
for j in range(0,c.shape[0]):
    c_l=c[j]
    for i in range(1,t.shape[0]):
        r_e_1[:,i]=((-r_e_1[:,i-1]+r_e_ss_1[:,i-1])/tau_e)*tstep+r_e_1[:,i-1]
        r_i_1[:,i]=((-r_i_1[:,i-1]+r_i_ss_1[:,i-1])/tau_i)*tstep+r_i_1[:,i-1]
        in_e_1[:,i]=c_l*h_1+np.dot(w_ee,r_e_1[:,i])-np.dot(w_ei,r_i_1[:,i])
        in_i_1[:,i]=c_l*h_1+np.dot(w_ie,r_e_1[:,i])-np.dot(w_ii,r_i_1[:,i])
        in_e_1[in_e_1[:,i]<0,i]=0
        in_i_1[in_i_1[:,i]<0,i]=0
        r_e_ss_1[:,i]=k*np.power(in_e_1[:,i],n)
        r_i_ss_1[:,i]=k*np.power(in_i_1[:,i],n)
    ext_in_1[:,j]=c_l*h_1 #external input
    R_e_1[:,j]=r_e_1[:,dt] #firing rate of E unit
    R_e_n_e_1[:,j]=np.dot(w_ee,r_e_1[:,dt]) #network E of E unit
    R_e_n_i_1[:,j]=np.dot(w_ei,r_i_1[:,dt]) #network I of E unit
    R_e_t_1[:,j]=in_e_1[:,dt] #I hope this is the net input of E unit
    R_i_1[:,j]=r_i_1[:,dt] #firing rate of I unit
    R_i_n_e_1[:,j]=np.dot(w_ie,r_e_1[:,dt]) #network E of I unit
    R_i_n_i_1[:,j]=np.dot(w_ii,r_i_1[:,dt]) #network I of I unit
    R_i_t_1[:,j]=in_i_1[:,dt] #I hope this is the net input of I unit
    sum_ext[j]=np.sum(c_l*h_1)
    sum_net_e_e[j]=np.sum(np.dot(w_ee,r_e_1[:,dt]))
    sum_net_e_i[j]=np.sum(np.dot(w_ei,r_i_1[:,dt]))
    sum_net_i_e[j]=np.sum(np.dot(w_ie,r_e_1[:,dt]))
    sum_net_i_i[j]=np.sum(np.dot(w_ii,r_i_1[:,dt]))

for j in range(0,c.shape[0]):
    c_l=c[j]
    for i in range(1,t.shape[0]):
        r_e_2[:,i]=((-r_e_2[:,i-1]+r_e_ss_2[:,i-1])/tau_e)*tstep+r_e_2[:,i-1]
        r_i_2[:,i]=((-r_i_2[:,i-1]+r_i_ss_2[:,i-1])/tau_i)*tstep+r_i_2[:,i-1]
        in_e_2[:,i]=c_l*h_2+np.dot(w_ee,r_e_2[:,i])-np.dot(w_ei,r_i_2[:,i])
        in_i_2[:,i]=c_l*h_2+np.dot(w_ie,r_e_2[:,i])-np.dot(w_ii,r_i_2[:,i])
        in_e_2[in_e_2[:,i]<0,i]=0
        in_i_2[in_i_2[:,i]<0,i]=0
        r_e_ss_2[:,i]=k*np.power(in_e_2[:,i],n)
        r_i_ss_2[:,i]=k*np.power(in_i_2[:,i],n)
    ext_in_2[:,j]=c_l*h_2 #external input
    R_e_2[:,j]=r_e_2[:,dt] #firing rate of E unit
    R_i_2[:,j]=r_i_2[:,dt] #firing rate of I unit

for j in range(0,c.shape[0]):
    c_l=c[j]
    for i in range(1,t.shape[0]):
        r_e_3[:,i]=((-r_e_3[:,i-1]+r_e_ss_3[:,i-1])/tau_e)*tstep+r_e_3[:,i-1]
        r_i_3[:,i]=((-r_i_3[:,i-1]+r_i_ss_3[:,i-1])/tau_i)*tstep+r_i_3[:,i-1]
        in_e_3[:,i]=c_l*h_1+c_l*h_2+np.dot(w_ee,r_e_3[:,i])-np.dot(w_ei,r_i_3[:,i])
        in_i_3[:,i]=c_l*h_1+c_l*h_2+np.dot(w_ie,r_e_3[:,i])-np.dot(w_ii,r_i_3[:,i])
        in_e_3[in_e_3[:,i]<0,i]=0
        in_i_3[in_i_3[:,i]<0,i]=0
        r_e_ss_3[:,i]=k*np.power(in_e_3[:,i],n)
        r_i_ss_3[:,i]=k*np.power(in_i_3[:,i],n)
    ext_in_3[:,j]=c_l*h_1+c_l*h_2 #external input
    R_e_3[:,j]=r_e_3[:,dt] #firing rate of E unit
    R_i_3[:,j]=r_i_3[:,dt] #firing rate of I unit