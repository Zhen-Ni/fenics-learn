#!/usr/bin/env python3

from collections import namedtuple
import numpy as np
from fenics import *
from ufl import nabla_div


W = 0.2
L = 3.0

Material = namedtuple('MaterialProperty', ['rho', 'E', 'mu'])
steel = Material(7800,210e9,0.3)
copper = Material(8500,110e9,0.32)

g=9.8

tol = 1e-6

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 30, 2, 2)
V = VectorFunctionSpace(mesh, 'P', 3)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= L / 2 + tol

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= L / 2 - tol

material_marker = MeshFunction('size_t', mesh, 3)
domain_left = Left()
domain_right = Right()
domain_left.mark(material_marker, 0)
domain_right.mark(material_marker, 1)


class MaterialLib(UserExpression):
    def __init__(self, name, material_marker=material_marker, steel=steel, copper=copper, **kwargs):
        self.name = name
        self.material_marker = material_marker
        self.steel = steel
        self.copper = copper
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        if self.material_marker[cell.index] == 0:
            values[0] = getattr(self.steel,self.name)
        else:
            values[0] = getattr(self.copper,self.name)

    def value_shape(self):
        return ()
    
E = MaterialLib('E', degree=2)
mu = MaterialLib('mu', degree=2)
rho = MaterialLib('rho', degree=2)



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

# Both are ok
#f = Constant((0, 0, -g))*rho
f = as_vector((0, 0, -g*rho))
T = Constant((0, 0, 0))

a = (np.dot(stress_normal(u,E,mu), strain_normal(v)) + 
     np.dot(stress_shear(u,E,mu), strain_shear(v)))*dx
L = dot(f, v)*dx + dot(T, v)*ds
     
     
tol = 1E-14
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)


# Compute solution
u = Function(V)
solve(a == L, u, bc)




V = FunctionSpace(mesh, 'P', 1)

# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
# plot(u_magnitude, 'Displacement magnitude')
print('min/max u:',
      u_magnitude.vector().min(),
      u_magnitude.vector().max())

# Save solution to file in VTK format
File('displacement1d-different-materials.pvd') << u
File('magnitude1d-different-materials.pvd') << u_magnitude




