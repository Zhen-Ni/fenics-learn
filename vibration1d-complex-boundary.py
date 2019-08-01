#!/usr/bin/env python3

import numpy as np
from fenics import *
from ufl import nabla_div

# Use Dirac Delta function for spring

E = 210e9
rho = 7800
mu = 0.3
W = 0.2
L = 3.0
g=9.8
k=Constant(1e8)


# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 120, 8, 8)
V = VectorFunctionSpace(mesh, 'P', 2)

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


f = Constant((0, 0, -rho*g))
T = Constant((0, 0, 0))

class Delta(UserExpression):
    def __init__(self, eps,x0, **kwargs):
        self.eps = eps
        self.x0 = x0
        UserExpression.__init__(self, **kwargs)
        
#    def eval(self, values, x):
#        eps = self.eps
#        x1 = np.array(x) - np.array(self.x0)
#        values[0] = eps/pi/(np.linalg.norm(x1)**2 + eps**2)
        
    def eval_cell(self, values, x, cell):
        eps = self.eps
        x1 = np.array(x) - np.array(self.x0)
        if abs(x1).max() <= eps:
            values[0] = 1.0/8/eps**3
        else:
            values[0] = 0


    def value_shape(self):
        return ()

delta = Delta(eps=5E-2, x0=(L,W,W),degree=5)
delta2 = Delta(eps=5E-2, x0=(L,0,W),degree=5)

ak = (np.dot(stress_normal(u,E,mu), strain_normal(v)) + 
     np.dot(stress_shear(u,E,mu), strain_shear(v)))*dx + 8*k*u[2]*v[2]*delta*dx+8*k*u[2]*v[2]*delta2*dx
am = rho * inner(u,v) * dx

tol = 1E-14
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

K = PETScMatrix()
M = PETScMatrix()

# Lead to asymmetric matrix
#assemble(ak,tensor=K)
#assemble(am,tensor=M)
#bc.apply(K)
#bc.apply(M)

dummy = v[0]*dx
assemble_system(ak,dummy,bc, A_tensor=K)
assemble_system(am, dummy,bc ,A_tensor=M)
bc.zero(M)



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


xdmffile = XDMFFile('vibration1d-complex-boundary.xdmf')
u = Function(V)
u.rename('u','uuuuu')
for i in range(20):
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    u.vector()[:] = rx
    xdmffile.write(u, r**.5/2/pi)
xdmffile.close()



