from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

T = 5.0            # final time
num_steps = 5000   # number of time steps
dt = T / num_steps # time step size
D = 0.01            # diffusion rate

# Create mesh
channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = generate_mesh(domain, 64)

# Define function spaces
C = FunctionSpace(mesh, 'P', 1)

# Define boundaries
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Define boundary conditions
bc = DirichletBC(C, Constant(1.0), cylinder)

# Define trial and test functions
c = TrialFunction(C)
cv = TestFunction(C)

# Define functions for solutions at previous and current time steps
c_n = Function(C)

# Define expressions used in variational forms
k  = Constant(dt)
D = Constant(D)

# Define variational problem for diffusion-convection equation
# F = ((c - c_n) / k)*cv*dx + dot(u_, grad(c))*cv*dx \
F = ((c - c_n) / k)*cv*dx + D*dot(grad(c), grad(cv))*dx
# F = c*cv*dx + k*dot(grad(c), grad(cv))*dx - c_n*cv*dx
a, L = lhs(F), rhs(F)

vtkfile_c = File('mix_ns_diff/c.pvd')

c = Function(C)

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Solve the diffusion-convection equation
    solve(a == L, c, bc)

    vtkfile_c << (c, t)

    # Update previous solution
    c_n.assign(c)
