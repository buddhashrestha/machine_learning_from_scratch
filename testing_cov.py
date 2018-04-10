import numpy as np

x = np.linspace(0,1,200)

y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]

x[100:200] = np.cos(4*np.pi*x)[100:200]

y = 4*np.linspace(0,1,200)+0.3*np.random.randn(200)

label= np.ones_like(x)

label[0:100]=0


z = zip(x,y)

utah = np.array(z).transpose()

print("cov : ",np.cov(utah))