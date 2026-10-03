import numpy as np
# Independent variable projection: scalar denominator search, exact real LS
# numerator at each trial. Uniform midpoint validation avoids fitting grid.
N=513
t=np.linspace(0,np.pi,N)
q=np.exp(-1j*t)
for name,dt,tf in [('first',.2,lambda s:1/(s+1)),('reduced',.2,lambda s:2/(s*s+3*s+2))]:
 h=tf(1j*t/dt)
 def objective(a):
  basis=np.stack([1/(1+a*q),q/(1+a*q)],axis=1)
  realbasis=np.vstack([basis.real,basis.imag]); rh=np.r_[h.real,h.imag]
  b=np.linalg.lstsq(realbasis,rh,rcond=None)[0]
  e=np.linalg.norm(basis@b-h)/np.linalg.norm(h)
  return e,b
 grid=np.linspace(-.99999,.99999,2001)
 center=grid[np.argmin([objective(a)[0] for a in grid])]
 left,right=max(-.99999,center-.003),min(.99999,center+.003)
 ratio=(np.sqrt(5)-1)/2
 for _ in range(80):
  x=right-ratio*(right-left);y=left+ratio*(right-left)
  if objective(x)[0]<objective(y)[0]:right=y
  else:left=x
 a=(left+right)/2
 b=objective(a)[1]
 tv=np.pi*(np.arange(4097)+.371)/4097
 qv=np.exp(-1j*tv)
 hv=tf(1j*tv/dt)
 prediction=(b[0]+b[1]*qv)/(1+a*qv)
 print(name,'den=',[1,a],'num=',b.tolist(),'RMS=',np.linalg.norm(prediction-hv)/np.linalg.norm(hv))
