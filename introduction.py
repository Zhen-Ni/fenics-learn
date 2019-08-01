#!/usr/bin/env python3


from fenics import *
import matplotlib.pyplot as plt

def approx(f, V):
    """Return Galerkin approximation to f in V."""
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u*v*dx
    L = f*v*dx
    u = Function(V)
    solve(a == L, u)
    return u

#class F(UserExpression):
#    def eval(self, values, x):
#        print(x)
#        values[0] = 2*x[0]*x[1] - x[0]*x[0]
        


def problem():
    f = Expression('2*x[0]*x[1] - pow(x[0], 2)', degree=2)
#    f = F()
    mesh = RectangleMesh(Point(0,-1), Point(2,1), 2, 2)
    V0 = FunctionSpace(mesh, 'DG', 0)
    u0 = approx(f, V0)
    u0.rename('u0', 'u0')
    u0_error = errornorm(f, u0, 'L2')
    V1 = FunctionSpace(mesh, 'P', 1)
    u1 = approx(f, V1)
    u1.rename('u1', 'u1')
    u1_error = errornorm(f, u1, 'L2')
    V2 = FunctionSpace(mesh, 'P', 2)
    u2 = approx(f, V2)
    u2.rename('u2', 'u2')
    u2_error = errornorm(f, u2, 'L2')
    print('L2 errors: e1=%g, e2=%g\n' % (u1_error, u2_error))
#     Simple plotting
    plt.figure()
    plot(f, title='f', mesh=mesh)
    plt.figure()
    plot(u0, title='u0')
    plt.figure()
    plot(u1, title='u1')
    plt.figure()
    plot(u2, title='u2')
    
if __name__ == '__main__':
    problem()
#    interactive() # Enable plotting