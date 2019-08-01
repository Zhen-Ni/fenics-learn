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
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 60, 4, 4)
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


f = Constant((0, 0, -rho*g))
T = Constant((0, 0, 0))

#spring_markers = MeshFunction('size_t', mesh, 0)
#class DomainSpring(SubDomain):
#    def inside(self, x, on_boundary):
#        return on_boundary and near(x[0],L) and near(x[1],W) and near(x[2],W)
#DomainSpring().mark(spring_markers,0)
#dp = Measure('dP', domain=mesh, subdomain_data=spring_markers)

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

a = (np.dot(stress_normal(u,E,mu), strain_normal(v)) + 
     np.dot(stress_shear(u,E,mu), strain_shear(v)))*dx + 8*k*u[2]*v[2]*delta*dx+8*k*u[2]*v[2]*delta2*dx
L = dot(f, v)*dx + dot(T, v)*ds

tol = 1E-14
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)


# Compute solution
u = Function(V)
solve(a == L, u, bc, solver_parameters={'linear_solver':'mumps'})



V = FunctionSpace(mesh, 'P', 1)

# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
# plot(u_magnitude, 'Displacement magnitude')
print('min/max u:',
      u_magnitude.vector().min(),
      u_magnitude.vector().max())

# Save solution to file in VTK format
u.rename('u','uu')
File('displacement1d-complex-boundary.pvd') << u
File('magnitude1d-complex-boundary.pvd') << u_magnitude

## Hold plot
#interactive()



