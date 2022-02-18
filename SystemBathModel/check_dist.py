#this is to check the initial distribution

import sys
import numpy as np

data=[]
for line in open('tmp.dat'):
	data.append(map(float,line.split()))

data = np.array(data,dtype=np.float64)
data = data.transpose()

Dim = 4
h = np.zeros((2*Dim,50),dtype=np.float64)
e = np.zeros((2*Dim,51),dtype=np.float64)
for i in range(Dim):
	h[i,:] , e[i,:] = np.histogram(data[i,:],bins=50,density=True)

for i in range(Dim):
	o = open('tmp.h.dat.'+str(i),'w')
	x = (e[i,0:-1] + e[i,1:]) / 2.
	y = h[i,:]
	for j in range(len(x)):
		o.write("%14.6g %14.6g\n" %(x[j],y[j]))
	o.close()
	
		
