#this is to check the initial distribution

import sys
import numpy as np

data=[]
for line in open('tmp.dat'):
	k = map(float,line.split())
	data.append(k[11]-k[2])

data = np.array(data,dtype=np.float64)

h , e = np.histogram(data,bins=50,density=True)
e = (e[0:-1]+e[1:])/2.

for i in range(len(h)):
	sys.stdout.write("%14.6g %14.6g\n" %(e[i],h[i]))
	
		
