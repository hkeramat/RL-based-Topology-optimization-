from fenics import *
from mshr import *
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

T = 5.0            # final time
num_steps = 5000   # number of time steps
dt = T / num_steps # time step size
mu = 0.001         # dynamic viscosity
rho = 1            # density
D = 0.01           # diffusion rate
save_each = 10     # number of time steps to skip for saving vtk results
plot_each = 20     # number of time steps to skip for saving plots

# Create mesh
channel = Rectangle(Point(0, 0), Point(2.0, 2.0)) # define the box
cylinder = Circle(Point(1.0, 0.4), 0.30) # define the internal obstacle
domain = channel - cylinder # create the domain by trimming the box
mesh = generate_mesh(domain, 64) # generate the mesh

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    # bottom boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary)

    # map top boundary (H) to bottom boundary (G)
    # map coordinates x in H to coordinates y in G
    def map(self, x, y):
        y[1] = x[1] - 2.0 # the dimension along y axis
        y[0] = x[0]

# Create periodic boundary condition
pbc = PeriodicBoundary()

# Define function spaces
C = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc) # for the temperature
V = VectorFunctionSpace(mesh, 'P', 2, constrained_domain=pbc) # for velocity (as a vector)
Q = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc) # for pressure

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.0)'
# walls    = 'near(x[1], 0) || near(x[1], 2.0)'
cylinder = 'on_boundary && x[0]>0.6 && x[0]<1.4 && x[1]>0.05 && x[1]<0.8' # internal obstacle

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(2.0 - x[1]) / pow(2.0, 2)', '0') # u-shape profile of the inlet velocity

# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow) # fixed velocity BC on inlet
# bcu_walls = DirichletBC(V, Constant((0, 0)), walls) # no_slip BC on walls
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder) # no_slip BC on obstacle
bcp_outflow = DirichletBC(Q, Constant(0), outflow) # zero pressure BC on outlet
bcu = [bcu_inflow, bcu_cylinder] # NS velcoity BCs
bcp = [bcp_outflow] # NS pressure BCs

bc_c = DirichletBC(C, Constant(300.0), cylinder) # constant temperate on obstacle
bc_in = DirichletBC(C, Constant(450.0), inflow) # constant temperate on inlet
bc = [bc_c, bc_in]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
c = TrialFunction(C)
cv = TestFunction(C)

# Define functions for solutions at previous and current time steps
u_n = Function(V) # implying zero initial conditions for velocities
u_  = Function(V)
p_n = Function(Q) # implying zero initial conditions for pressure
p_  = Function(Q)
c_n = interpolate(Expression("273.0", degree=2), C) # initial conditions for temperature

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
[bc.apply(A1) for bc in bcu] # velocity BCs
[bc.apply(A2) for bc in bcp] # pressure BCs

# Create XDMF files for visualization output
xdmffile_u = XDMFFile('output/velocity.xdmf') # velocity output file
xdmffile_p = XDMFFile('output/pressure.xdmf') # pressure output file
vtkfile_c = File('output/c.pvd') # temperature vtk output file

# allow XDMF files be written after each step
xdmffile_u.parameters["flush_output"] = True
xdmffile_p.parameters["flush_output"] = True

# Create progress bar
progress = Progress('Looping', num_steps)
# set_log_level(LogLevel.PROGRESS)

# create figs directory for saving matplotlib plots
if not os.path.isdir("figs"):
    os.makedirs("figs")

c = Function(C)

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu] # apply velocity BCs
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg') # solve velocity equation

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp] # apply pressure BCs
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg') # solve pressure equation

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor') # solve coupling equation

    # Solve the diffusion-convection equation
    solve(a4 == L4, c, bc)

    # Save solution to file
    if (n % save_each == 0): # if it's time to write output:
        xdmffile_u.write(u_, t)
        xdmffile_p.write(p_, t)
        vtkfile_c << (c, t)

    if (n % plot_each == 0): # if it's time to plot pressure and temperature profiles
        # plot pressure profile on inlet
        tol = 0.001  # avoid hitting points outside the domain
        y = np.linspace(0 + tol, 2.0 - tol, 101) # generate y value of points at which evaluation occurs
        points = [(0, y_) for y_ in y] # generate the set of points in the inlet
        p_line = np.array([p_(point) for point in points]) # evaluate the value of pressure on generated points
        plt.plot(y, p_line, 'b', linewidth=2) # plot pressure profile in blue
        plt.xlabel('$y$')
        plt.savefig(f'figs/p_curve-{n}.png') # save the plot
        plt.clf()

        # plot temperate profile on outlet
        points = [(2.0, y_) for y_ in y] # generate the set of points in the outlet
        t_line = np.array([c(point) for point in points]) # evaluate the value of temperature on generated points
        plt.plot(y, t_line, 'r', linewidth=2) # plot temperature profile in red
        plt.xlabel('$y$')
        plt.savefig(f'figs/t_curve-{n}.png') # save the plot
        plt.clf()

        # compute and print the weighted average value of pressure (inlet) and temperature (outlet)
        p_average = np.average(p_line)
        c_average = np.average(t_line)
        print(f"=====> Averaged pressure at inlet: {p_average} <=====")
        print(f"=====> Averaged temperate at outlet: {c_average} <=====")


    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    c_n.assign(c)

    # Update progress bar
    progress += 1
    print('u max:', u_.vector().max())
