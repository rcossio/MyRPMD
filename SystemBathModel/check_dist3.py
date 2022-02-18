from scipy.sparse import diags
import numpy as np
import sys


#----------------------------------
#	Renaming math functions
#----------------------------------
pi  = np.pi
exp = np.exp
log = np.log10
sqrt = np.sqrt
tanh = np.tanh
cosh = np.cosh

#----------------------------------
#	Parameters
#----------------------------------
T    = 1000.0                 # in K
Kb   = 1.3806488e-23          # in J/K 
V0   = 0.425*1.60218e-19      # in J
m    = 1061.*9.10938356e-31   # in kg
a    = 0.734*5.2918e-11       # in m
dt   = 20*1e-18               # in s
hbar = 1.0545718e-34          # in m**2*kg/s
w_c  = 500 *1e2               # in 1/m
w_b  = 500 *1e2               # in 1/m
V0pp = 2085*1e2		      # in 1/m
nsamples = 1000000
Nbids    = 4
f=3

#---------------------------------
#	Constants
#---------------------------------
beta = 1/(Kb*T)
beta_n = beta/Nbids
w_n = 1/(beta_n*hbar)
sigmap= sqrt(m/beta_n)
S_T = sqrt(2*pi*beta*hbar**2/m)
Q = 1./S_T
N_p = 1./S_T
eta = 1.0*m*w_b

#--------------------------------
#	Functions
#--------------------------------
def w(i):
        return w_c*np.log((i+1-1.5)/float(f-1))

def c(i):
        return w(i)*np.sqrt((2*eta*m*w_c)/(np.pi*(f-1)))

def calcForce (q):
        F = np.zeros((f,Nbids),dtype=np.float64)
	F[0,:] = 0.5* m* w_b**2 *q[0,:] - m**2 *w_b**4* q[0,i]**3/(4*V0pp)
	for k in range(1,f):
		F[k,:] = -m* w(k)**2 *q[k,:] +c(k)*q[0,:]
	return F

def calc_derivada_p (q):
	derivada = calcForce(q)
	for j in range(Nbids):
		if   ( j == 0 ):
                        derivada[:,j] += -m*w_n**2*(2*q[:,j]-q[:,Nbids-1]-q[:,j+1])
		elif ( j == Nbids-1):
                        derivada[:,j] += -m*w_n**2*(2*q[:,j]-q[:,j-1]    -q[:,0]  )
		else:
			derivada[:,j] += -m*w_n**2*(2*q[:,j]-q[:,j-1]    -q[:,j+1])
	return derivada
	
	
def calcPotential (qq):
	q = C.T.dot(qq)
	V = 0.0
	for i in range(Nbids):
		V += -0.5* m* w_b**2 *q[0,i]**2 + m**2 *w_b**4* q[0,i]**4/(16*V0pp)
		for k in range(1,f):
			V +=  0.5* m* w(k)**2 *(q[k,i]-c(k)*q[0,i]/(m*w(k)**2))**2
	return V

def deltaV(q):
	deltav = calcPotential(q)
	for k in range(1,f):
		deltav -= 0.5 * m * w_nm[k]**2 * np.linalg.norm(q[i,:])**2
	return deltav

def heaviside(q):
	qq = C.dot(q)
        return np.heaviside(np.mean(qq[0,:]), 1.0)

def report(string):
        output = open(outfile,'a')
        output.write(string)
        output.close()

#-----------------------------------------
#	Normal mode transformation
#-----------------------------------------
Hessian = np.zeros((f,f),dtype=np.float64)
Hessian[0,0] = -m*w_b**2
for i in range(1,f):
	Hessian[0,i] = -c(i)
	Hessian[i,0] = -c(i)
	Hessian[i,i] = m*w(i)**2
evals, C = np.linalg.eigh(Hessian) 

evals[1:] = evals[1:][::-1]
C[:,1:] = C[:,1:][:,::-1]
w_nm=np.sqrt(evals/m)

#get points
v= 1/beta_n/m/w_n**2
for i in range(4000):
        x = -2e-10+1e-13*i
        y = np.exp(-x**2/(2*v),dtype=np.float64)/np.sqrt(2*np.pi*v)
        print "%14.6g %14.6g" %(x,y)

