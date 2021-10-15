from fenics import *
from mshr import *
import numpy as np

T = 5.0            # final time
num_steps = 5000   # number of time steps
dt = T / num_steps # time step size
mu = 0.001         # dynamic viscosity
rho = 1            # density
D = 0.01           # diffusion rate
save_each = 10     # number of time steps to skip when saving

# Create mesh
channel = Rectangle(Point(0, 0), Point(2.0, 2.0))
cylinder = Circle(Point(1.0, 1.0), 0.20)
domain = channel - cylinder
mesh = generate_mesh(domain, 64)

# Define function spaces
C = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.0)'
walls    = 'near(x[1], 0) || near(x[1], 2.0)'
cylinder = 'on_boundary && x[0]>0.6 && x[0]<1.4 && x[1]>0.6 && x[1]<1.4'

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(2.0 - x[1]) / pow(2.0, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]
bc = DirichletBC(C, Constant(1.0), cylinder)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
c = TrialFunction(C)
cv = TestFunction(C)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)
c_n = Function(C)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)
D = Constant(D)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Define variational problem for diffusion-convection equation
F = ((c - c_n) / k)*cv*dx + dot(u_, grad(c))*cv*dx \
  + D*dot(grad(c), grad(cv))*dx
a4 = lhs(F)
L4 = rhs(F)

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Create XDMF files for visualization output
xdmffile_u = XDMFFile('output/velocity.xdmf')
xdmffile_p = XDMFFile('output/pressure.xdmf')
vtkfile_c = File('output/c.pvd')

xdmffile_u.parameters["flush_output"] = True
xdmffile_p.parameters["flush_output"] = True

# Create progress bar
progress = Progress('Looping', num_steps)
set_log_level(LogLevel.PROGRESS)

c = Function(C)

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Solve the diffusion-convection equation
    solve(a4 == L4, c, bc)

    # Save solution to file
    if (n % save_each == 0):
        xdmffile_u.write(u_, t)
        xdmffile_p.write(p_, t)
        vtkfile_c << (c, t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    c_n.assign(c)

    # Update progress bar
    progress += 1
    print('u max:', u_.vector().max())
