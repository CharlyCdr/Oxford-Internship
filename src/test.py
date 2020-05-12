import numpy as np
from network import Dynamics as dyn
from network import Economy as eco
import plotly.graph_objs as go
import tqdm as tqdm
import plotly.offline as plo

n = 100
d = 3
alpha = 0.1
beta = 0.05
z = np.ones(n)
theta = np.ones(n)/n
V = np.ones(n)
eps = 1

# eig_s = np.array([])
# for k in tqdm.tqdm((range(10))):
#     test = eco(n=n, d=d, z=z, J=1, V=V, theta=theta)
#     test.gen_M(eps=eps, directed=True, ntype='m_regular')
#     test.compute_p()
#     test.compute_lda()
#     test.set_Sd(alpha=alpha, beta=beta, q=.5)
#     eig_s = np.append(eig_s, np.linalg.eigvals(test.Sd))
#
# eig_s = np.array(eig_s)
# fig = go.Figure()
# fig.add_trace(go.Scattergl(x=np.real(eig_s), y=np.imag(eig_s), mode='markers'))
# plo.plot(fig, filename='test.html')


sim = dyn(n=n)
sim.set_eco(d=d, z=z, J=1, V=V, theta=theta, eps=eps, directed=True, ntype='m_regular')


x0 = np.random.uniform(1,10,2*n)
x1 = x0
end = 1000

pr = sim.surplus_dyn(end, x0, x1, alpha, beta, q=1, cont=False)


