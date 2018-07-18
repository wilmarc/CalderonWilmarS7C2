import numpy as np
import matplotlib.pyplot as plt


data=np.genfromtxt("CircuitoRC.txt")
t=data[:,0]
q=data[:,1]
Vo=10
n=10000
l=np.zeros(n)
r=np.zeros(n)
c=np.zeros(n)

def carga(t,r,c):
	qmax=Vo*c
	return qmax*(1-np.exp((-1*t)/(r*c)))

def like(qes,qdad):
	suma=0.0
	for i in range(len(qes)):
		suma+=(qes[i]-qdad[i])**2
	suma=suma/n
	return np.exp((-0.5)*suma)
def mhm():
	for i in range(n):
		rguess=20.0*np.random.random()
		cguess=20.0*np.random.random()
		rnew=np.random.normal(rguess,0.1)
		cnew=np.random.normal(cguess,0.1)
		q1=carga(t,rguess,cguess)
		q2=carga(t,rnew,cnew)
		lprimero=like(q1,q)
		lprima=like(q2,q)
		al=(lprima/lprimero)
		if(al>=1.0):
			rin=rnew
			cin=cnew
			l[i]+=lprima
		else:
			bet=np.random.random()
			if(al>bet):
				rin=rnew
				cin=cnew
				l[i]+=lprima
			else:
				rin=rguess
				l[i]+=lprimero
				cin=cguess
		r[i]=rin
		c[i]=cin
mhm()		
indice=np.argmax(l)
rbest=r[indice]
cbest=c[indice]
print (indice,rbest,cbest)
plt.figure()
plt.scatter(t,q,s=2)
plt.plot(t,carga(t,rbest,cbest))
plt.savefig("CargaRC.pdf")
#plt.show()
