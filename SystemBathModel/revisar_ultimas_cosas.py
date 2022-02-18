from scipy.sparse import diags
import numpy as np
import sys


#----------------------------------
#	Renaming math functions
#----------------------------------
pi  = np.pi
sin = np.sin
exp = np.exp
log = np.log10
sqrt = np.sqrt
tanh = np.tanh
cosh = np.cosh

#--------------------------------
#	Functions
#--------------------------------
def w(i):
        return w_c*np.log((i+1-1.5)/float(f-1))

def c(i):
        return w(i)*np.sqrt((2*eta*m*w_c)/(np.pi*(f-1)))

def calcForce (q):
        F = np.zeros((f,Nbids),dtype=np.float64)
	F[0,:] = m* w_b**2 *q[0,:] - m**2 *w_b**4* q[0,:]**3/(4*V0pp) 
	for k in range(1,f):
		F[0,:] += c(k)*(q[k,:]-c(k)*q[0,:]/m/w(k)**2) 
		F[k,:]  = -m* w(k)**2 *q[k,:] +c(k)*q[0,:]

#	Esto es una mentira: pongo una fuerza armonica para ver como se mueve el polimero
#	for u in range(f):
#		for v in range(Nbids):
#			F[u,v] = - m* w(1)**2 *q[u,v]
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
	
	
def calcPotential (q):
	V = 0.0
	for i in range(Nbids):
		V += -0.5* m* w_b**2 *q[0,i]**2 + m**2 *w_b**4* q[0,i]**4/(16*V0pp)
		for k in range(1,f):
			V +=  0.5* m* w(k)**2 *(q[k,i]-c(k)*q[0,i]/(m*w(k)**2))**2

#       Esto es una mentira: pongo una fuerza armonica para ver como se mueve el polimero
#        for v in range(Nbids):
#                for u in range(f):
#                        V +=  0.5* m* w(1)**2 *q[u,v]**2

	return V

def extendedPotential(q):
	H = calcPotential(q)
	for k in range(f):
	        for j in range(Nbids):
	                if   ( j == 0 ):
	                        H += 0.5*m*w_n**2*(q[k,j]-q[k,Nbids-1])**2
	                else:
	                        H += 0.5*m*w_n**2*(q[k,j]-q[k,j-1])**2
	return H


def deltaV(q):
	deltav = calcPotential(q)
	for k in range(1,f):
		deltav -= 0.5 * m * w_nm[k]**2 * np.linalg.norm(q[k,:])**2
	return deltav

def heaviside(q):
	qq = C.T.dot(q)
        return np.heaviside(np.mean(qq[0,:]), 1.0)

def heaviside_n(q):
        qq = C.T.dot(q)
        return np.mean(np.heaviside(qq[0,:], 1.0))

def calc_q_n(w):
        q_n =1.0
        for l in range(1,Nbids+1):
		q_n /= sqrt(4*sin(l*pi/Nbids)**2+(beta*hbar*w/Nbids)**2)
	return q_n

for line in open('results.dat'):
	mweighted    = float(line.split()[3])
        eta_over_mwb = float(line.split()[1])

	#----------------------------------
	#       Parameters
	#----------------------------------
	T    = 300.0                          # in K
	Kb   = 3.1672083472e-06               # in Eh/K 
	m    = 1061.0                         # in m_e
	a    = 0.734                          # in a0
	dt   = 48e0                            # in Eh/hbar   con 47-48 son 1.15fs  originalmente con 8 va re bien
	hbar = 1.0                            # in hbar
	w_c  = 500  *4.5454545454545455e-06   # in Eh
	w_b  = 500  *4.5454545454545455e-06   # in Eh
	V0pp = 2085 *4.5454545454545455e-06   # in Eh

	nsamples = 5000
	nsteps   = 250
	Nbids    = 4
	f        = 10


	#---------------------------------
	#       Constants
	#---------------------------------
	beta = 1/(Kb*T)
	beta_n = beta/Nbids
	w_n = 1/(beta_n*hbar)
	sigmap= sqrt(m/beta_n)
	S_T = sqrt(2*pi*beta*hbar**2/m)
	eta = eta_over_mwb*m*w_b
	Freq = 10

	#-----------------------------------------
	#	Normal mode transformation
	#-----------------------------------------
	Hessian = np.zeros((f,f),dtype=np.float64)
	Hessian[0,0] = -m*w_b**2
	for i in range(1,f):
		Hessian[0,i] = -c(i)
		Hessian[i,0] = -c(i)
		Hessian[i,i] = m*w(i)**2
		Hessian[0,0] += c(i)**2/m/w(i)**2
	evals, C = np.linalg.eigh(Hessian) 

	evals[1:] = evals[1:][::-1]
	C[:,1:] = C[:,1:][:,::-1]
	if C[0,0] < 0.0:
		C = -C
	w_nm=np.sqrt(evals/m)

	#-----------------------------------------
	#	Calculating Np
	#-----------------------------------------
	Np = 1/S_T
	for k in range(1,f):
		Np *= calc_q_n(w_nm[k])

	#-----------------------------------------------------
	#	Calculating Qr (previo normal mode analysis)
	#-----------------------------------------------------
	HessianR = np.zeros((f,f),dtype=np.float64)
	HessianR[0,0] = 2*m*w_b**2
	for i in range(1,f):
	        HessianR[0,i] = -c(i)
	        HessianR[i,0] = -c(i)
	        HessianR[i,i] = m*w(i)**2
	        HessianR[0,0] += c(i)**2/m/w(i)**2
	evals, Cr = np.linalg.eigh(HessianR)

	evals = evals[::-1]
	Cr = Cr[:,::-1]
	w_nm_reactivo=np.sqrt(evals/m)
	Qr = exp(beta*V0pp)
	for k in range(f):
	        Qr *= calc_q_n(w_nm_reactivo[k])

	#---------------------------------------------
	#	Find transmission coefficient
	#---------------------------------------------
	C_t = Np * mweighted
	k_t = C_t / Qr
	k_CL = w_b*exp(-beta*V0pp)/pi/sqrt(2)
	print "%14.6g %14.6g %14.6g " %(T,eta_over_mwb,k_t/k_CL)

