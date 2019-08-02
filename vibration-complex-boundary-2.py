#!/usr/bin/env python3

import numpy as np
from fenics import *
from ufl import nabla_div

# Add spring in the stiffness matrix directly

E = 210e9
rho = 7800
mu = 0.3
W = 0.2
L = 3.0
g=9.8
k=1e8


# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 15, 1, 1)
V = VectorFunctionSpace(mesh, 'P', 3)

def strain_normal(u):
    return u[0].dx(0), u[1].dx(1), u[2].dx(2)

def strain_shear(u):
    return u[1].dx(2)+u[2].dx(1), u[2].dx(0)+u[0].dx(2), u[0].dx(1)+u[1].dx(0)
    
def stress_normal(u, E, mu):
    epsilon = strain_normal(u)
    coeff = E / (1+mu)/(1-2*mu)
    sigma0 = (1-mu) * epsilon[0] + mu * epsilon[1] + mu * epsilon[2]
    sigma1 = mu * epsilon[0] + (1-mu) * epsilon[1] + mu * epsilon[2]
    sigma2 = mu * epsilon[0] + mu * epsilon[1] + (1-mu) * epsilon[2]
    return coeff * sigma0, coeff * sigma1, coeff * sigma2 

def stress_shear(u, E, mu):
    epsilon = strain_shear(u)
    coeff = E / (1+mu)/(1-2*mu)
    sigma0 = 0.5 * (1-2*mu) * epsilon[0]
    sigma1 = 0.5 * (1-2*mu) * epsilon[1]
    sigma2 = 0.5 * (1-2*mu) * epsilon[2]
    return coeff * sigma0, coeff * sigma1, coeff * sigma2 
    
u = TrialFunction(V)
v = TestFunction(V)



ak = (np.dot(stress_normal(u,E,mu), strain_normal(v)) + 
     np.dot(stress_shear(u,E,mu), strain_shear(v)))*dx
am = rho * inner(u,v) * dx

tol = 1E-14
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

K = PETScMatrix()
M = PETScMatrix()


dummy = v[0]*dx
assemble_system(ak,dummy,bc, A_tensor=K)
assemble_system(am, dummy,bc ,A_tensor=M)
bc.zero(M)

dof_spring = []
for i,(x,y,z) in enumerate(V.tabulate_dof_coordinates()):
    if near(x,L) and near(y,W) and near(z,W):
        dof_spring.append(i)
    if near(x,L) and near(y,0) and near(z,W):
        dof_spring.append(i)

dof_spring = dof_spring[2::3]

k_old = np.array([[0.0]])
for i in dof_spring:
    K.get(k_old,[i],[i])
    k_new = k + k_old[0,0]
    K.set([k_new], [i],[i])
    K.apply("add")


eigensolver = SLEPcEigenSolver(K,M)
eigensolver.parameters["spectrum"] = "target magnitude"
eigensolver.parameters["spectral_transform"] = "shift-and-invert"
eigensolver.parameters["spectral_shift"] = 0.0
eigensolver.solve(60)



neigs = 50
computed_eigenvalues = []
for i in range(neigs):
    r, _ = eigensolver.get_eigenvalue(i) # ignore the imaginary part
    computed_eigenvalues.append(r**.5/2/pi)
print(np.sort(np.array(computed_eigenvalues)))


xdmffile = XDMFFile('vibration1d-complex-boundary-2.xdmf')
u = Function(V)
u.rename('u','uuuuu')
for i in range(20):
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    u.vector()[:] = rx
    xdmffile.write(u, r**.5/2/pi)
xdmffile.close()



