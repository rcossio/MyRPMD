#python this.py output
#grep -v '#' nbids.2000.dat | awk '{n=n+1; S=S+$1}END{print S/n}'
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
T    = 2000.0                 # in K
Kb   = 1.3806488e-23          # in J/K 
V0   = 0.425*1.60218e-19      # in J
m    = 1061.*9.10938356e-31   # in kg
a    = 0.734*5.2918e-11       # in m
dt   = 20*1e-18                # in s
hbar = 1.0545718e-34          # in m**2*kg/s

nsamples = 500
Nbids    = 64
outfile  = sys.argv[1]

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


#--------------------------------
#	Functions
#--------------------------------
def calcForce (q):
	num = 2*V0*tanh(q/a)
	den = a * (cosh(q/a))**2
	return num/den

def calc_derivada_p (q):
	derivada = calcForce(q)
	for j in range(Nbids):
		if   ( j == 0 ):
                        derivada[j] += -m*w_n**2*(2*q[j]-q[Nbids-1]-q[j+1])
		elif ( j == Nbids-1):
                        derivada[j] += -m*w_n**2*(2*q[j]-q[j-1]    -q[0]  )
		else:
			derivada[j] += -m*w_n**2*(2*q[j]-q[j-1]    -q[j+1])

	return derivada
	
	
def calcPotential (q):
	return np.sum(V0/(cosh(q/a))**2)

def heaviside_n(q):
        return np.mean(np.heaviside(q, 1.0))

def report(string):
        output = open(outfile,'a')
        output.write(string)
        output.close()

#-----------------------------------
#	Cholesky matrix
#-----------------------------------
Ndim = Nbids-1
tmp = diags([-1, 2, -1], [-1, 0, 1], shape=(Ndim, Ndim)).toarray()
Qcov = np.linalg.inv(tmp)
L = np.linalg.cholesky(Qcov)
L /= L[0,0]
L /= np.sqrt(beta_n*m*w_n**2)

#-------------------------------
#	Initial sampling
#-------------------------------
z   = []
v_s = []
bf  = []
weighted = np.zeros(nsamples,dtype=np.float64)

for s in range(nsamples/2):
	#------------------------------
	#	Random sampling
	#------------------------------
	r = np.random.normal(loc=0.0,scale=1.0,size=L.shape[0])
	q = np.zeros(Nbids,dtype=np.float64)
	q[1:Nbids] = L.dot(r) 
	p = np.random.normal(loc=0.0,scale=sigmap,size=Nbids)
	
	#-------------------------------
	#	Sampling	
	#-------------------------------
	z.append((q,p))
	v_s.append(p[0]/m)
        deltav = calcPotential(q)
        bf.append( exp(-beta_n*deltav))

	#-------------------------------
	#	Symmetric Sampling
	#------------------------------
	p = -p
        z.append((q,p))
        v_s.append(p[0]/m)
        deltav = calcPotential(q)
        bf.append( exp(-beta_n*deltav))

v_s = np.array(v_s, dtype=np.float64)
bf  = np.array(bf,  dtype=np.float64)


#------------------------------------
#	Trajectories
#------------------------------------
for s in range(nsamples):
	h_n = 0.5
	t = 0
	while 0.0+1e-12 < h_n < 1.0-1e-12:
		
		#---------------------------------------------------
		#	Initial state
		#---------------------------------------------------
		if t == 0:
			(q,p) = z[s]

		#---------------------------------------------------
		#	Symplectic integrator - velocity verlet
		#---------------------------------------------------
		derivada_p = calc_derivada_p(q)
		p = p + 0.5 * dt * derivada_p
		q = q + 1.0 * dt * p/m
		derivada_p = calc_derivada_p(q)
		p = p + 0.5 * dt * derivada_p
		t += 1

		#---------------------------------------------------
		#	h_n(t) calculation
		#---------------------------------------------------
		if t%10 == 0:
			h_n = heaviside_n(q)

	#---------------------------------------------------------
	#	Find wight h_n
	#---------------------------------------------------------
        weighted [s] = bf[s] * v_s[s] * h_n
	report("%18.10g \n" %weighted[s])

#---------------------------------------------
#	Find transmission coefficient
#---------------------------------------------
C_t = N_p * np.mean (weighted)
k_t = C_t / Q
report("# %14.6g %14.6g %14.6g %14.6g \n" %(T,1000/T,k_t,log(k_t)))

