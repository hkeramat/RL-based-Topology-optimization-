# Generic imports
import os
import sys
import math
#import time
import numpy               as np
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec

# Custom imports
from dolfin    import *
from mshr      import *

n_call = 0
u_0    = None
V_0    = None

def solve_flow(*args, **kwargs):
    # Handle optional arguments
    mesh_file   = kwargs.get('mesh_file',   'shape.xml')
    output      = kwargs.get('output',      False)
    final_time  = kwargs.get('final_time',  15.0)
    reynolds    = kwargs.get('reynolds',    10.0)
    pts_x       = kwargs.get('pts_x',       np.array([]))
    pts_y       = kwargs.get('pts_y',       np.array([]))
    cfl         = kwargs.get('cfl',         0.5)
    xmin        = kwargs.get('xmin',       -15.0)
    xmax        = kwargs.get('xmax',        30.0)
    ymin        = kwargs.get('ymin',       -15.0)
    ymax        = kwargs.get('ymax',        15.0)

    # Parameters
    v_in      = 1.0
    mu        = 1.0/reynolds
    rho       = 1.0
    tag_shape = 5
    x_shape   = 4.0
    y_shape   = 4.0
    t_in      = 450.0
    D         = 1.0
    t_init    = 273.0
    t_obs     = 300.0
    sol_file  = 'shape.pvd'

    # Create subdomain containing shape boundary
    class Obstacle(SubDomain):
        def inside(self, x, on_boundary):
            return (on_boundary and
                    (-x_shape < x[0] < x_shape) and
                    (-y_shape < x[1] < y_shape))

    # Sub domain for Periodic boundary condition
    class PeriodicBoundary(SubDomain):
        # bottom boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool((x[1]-ymin) < DOLFIN_EPS and (x[1]-ymin) > -DOLFIN_EPS and on_boundary)

        # map top boundary (H) to bottom boundary (G)
        # map coordinates x in H to coordinates y in G
        def map(self, x, y):
            y[1] = x[1] - (ymax-ymin) # the dimension along y axis
            y[0] = x[0]

    # Create periodic boundary condition
    pbc = PeriodicBoundary()

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2.0*mu*epsilon(u) - p*Identity(len(u))

    # # Compute drag and lift
    # def compute_drag_lift(u, p, mu, normal, gamma):
    #     eps      = 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #     sigma    = 2.0*mu*eps - p*Identity(len(u))
    #     traction = dot(sigma, normal)
    #
    #     forceX   = traction[0]*gamma
    #     forceY   = traction[1]*gamma
    #     fX       = assemble(forceX)
    #     fY       = assemble(forceY)
    #
    #     return (fX, fY)

    # Create mesh
    # Ugly hack : change dim=3 to dim=2 in xml mesh file
    os.system("sed -i 's/dim=\"3\"/dim=\"2\"/g' "+mesh_file)
    mesh = Mesh(mesh_file)
    h    = mesh.hmin()

    # Compute timestep and max nb of steps
    dt        = cfl*h/v_in
    timestep  = dt
    T         = final_time
    num_steps = math.floor(T/dt)
    print(num_steps)

    # Define output solution file
    vtkfile = File(sol_file)

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2, constrained_domain=pbc)
    Q = FunctionSpace      (mesh, 'P', 1, constrained_domain=pbc)
    C = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc) # for the temperature

    # Define boundaries
    inflow  = 'near(x[0], '+str(math.floor(xmin))+')'
    outflow = 'near(x[0], '+str(math.floor(xmax))+')'
    # wall1   = 'near(x[1], '+str(math.floor(ymin))+')'
    # wall2   = 'near(x[1], '+str(math.floor(ymax))+')'
    shape   = 'on_boundary && x[0]>(-'+str(x_shape)+') && x[0]<'+str(x_shape)+' && x[1]>(-'+str(y_shape)+') && x[1]<('+str(y_shape)+')'

    # Define boundary conditions
    bcu_inflow  = DirichletBC(V,        Constant((v_in, 0.0)), inflow)
    # bcu_wall1   = DirichletBC(V.sub(1), Constant(0.0),         wall1)
    # bcu_wall2   = DirichletBC(V.sub(1), Constant(0.0),         wall2)
    bcu_aile    = DirichletBC(V,        Constant((0.0, 0.0)),  shape)
    bcp_outflow = DirichletBC(Q,        Constant(0.0),         outflow)
    # bcu         = [bcu_inflow, bcu_wall1, bcu_wall2, bcu_aile]
    bcu         = [bcu_inflow, bcu_aile]
    bcp         = [bcp_outflow]

    bc_c = DirichletBC(C, Constant(t_obs), shape) # constant temperate on obstacle
    bc_in = DirichletBC(C, Constant(t_in), inflow) # constant temperate on inlet
    bc = [bc_c, bc_in]

    # Tag shape boundaries for drag_lift computation
    obstacle    = Obstacle()
    boundaries  = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    obstacle.mark(boundaries, tag_shape)
    ds          = Measure('ds', subdomain_data=boundaries)
    gamma_shape = ds(tag_shape)

    # Define trial and test functions
    u, v  = TrialFunction(V), TestFunction(V)
    p, q  = TrialFunction(Q), TestFunction(Q)
    c, cv = TrialFunction(C), TestFunction(C)

    # Define functions for solutions at previous and current time steps
    u_n, u_, u_m = Function(V), Function(V), Function(V)
    p_n, p_      = Function(Q), Function(Q)
    c_n = interpolate(Expression(f"{t_init}", degree=2), C) # initial conditions for temperature

    # Define initial value
    global n_call
    global u_0

    if (n_call != 0):
        u_0.set_allow_extrapolation(True)
        u_n  = project(u_0, V)
        show = True
    else:
        show = False

    # Define expressions and constants used in variational forms
    U   = 0.5*(u_n + u)
    n   = FacetNormal(mesh)
    f   = Constant((0, 0))
    dt  = Constant(dt)
    mu  = Constant(mu)
    rho = Constant(rho)
    D = Constant(D)

    # Set BDF2 coefficients for 1st timestep
    bdf2_a = Constant( 1.0)
    bdf2_b = Constant(-1.0)
    bdf2_c = Constant( 0.0)

    # Define variational problem for step 1
    # Using BDF2 scheme
    F1 = dot((bdf2_a*u + bdf2_b*u_n + bdf2_c*u_m)/dt, v)*dx + dot(dot(u_n, nabla_grad(u)), v)*dx + inner(sigma(u, p_n), epsilon(v))*dx + dot(p_n*n, v)*ds - dot(mu*nabla_grad(u)*n, v)*ds - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p),   nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (bdf2_a/dt)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u,  v)*dx
    L3 = dot(u_, v)*dx - (dt/bdf2_a)*dot(nabla_grad(p_ - p_n), v)*dx

    # Define variational problem for diffusion-convection equation
    F = ((c - c_n) / dt)*cv*dx + dot(u_, grad(c))*cv*dx \
      + D*dot(grad(c), grad(cv))*dx
    a4 = lhs(F)
    L4 = rhs(F)

    # Assemble A3 matrix since it will not need re-assembly
    A3 = assemble(a3)

    # # Initialize drag and lift
    # drag      = 0.0
    # lift      = 0.0
    # lfdr      = 0.0
    # drag_inst = np.array([])
    # lift_inst = np.array([])
    # lfdr_inst = np.array([])
    # drag_avg  = np.array([])
    # lift_avg  = np.array([])
    # lfdr_avg  = np.array([])

    pressure_drop = 0.0
    heat = 0.0

    ########################################
    # Time-stepping loop
    ########################################
    try:
        k     = 0
        t     = 0.0
        t_arr = np.array([])
        c = Function(C)

        set_log_active(False)

        for m in range(num_steps):
            # Update current time
            t += timestep

            # Step 1: Tentative velocity step
            A1 = assemble(a1)
            b1 = assemble(L1)
            [bc.apply(A1) for bc in bcu]
            [bc.apply(b1) for bc in bcu]
            solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg') #gmres

            # Step 2: Pressure correction step
            A2 = assemble(a2)
            b2 = assemble(L2)
            [bc.apply(A2) for bc in bcp]
            [bc.apply(b2) for bc in bcp]
            solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

            # Step 3: Velocity correction step
            b3 = assemble(L3)
            solve(A3, u_.vector(), b3, 'cg',       'sor')

            solve(a4 == L4, c, bc)

            # Update previous solution
            u_m.assign(u_n)
            u_n.assign(u_)
            p_n.assign(p_)
            c_n.assign(c)

            # vtkfile << (c, t)


            # Set BDF2 coefficients for m>1
            bdf2_a.assign(Constant( 3.0/2.0))
            bdf2_b.assign(Constant(-2.0))
            bdf2_c.assign(Constant( 1.0/2.0))

#                # Check execution time
#                end     = time.time()
#                elapsed = end - start

#            if (elapsed > max_time):
#                return 0.0, 0.0, False
            ########################################


        filename = mesh_file.split('_')[-1]
        filename = filename.split('.')[0]

        # plot pressure profile on inlet
        plt.clf()
        tol = 0.001  # avoid hitting points outside the domain
        y = np.linspace(ymin + tol, ymax - tol, 101) # generate y value of points at which evaluation occurs
        points = [(xmin, y_) for y_ in y] # generate the set of points in the inlet
        p_line = np.array([p_(point) for point in points]) # evaluate the value of pressure on generated points
        plt.plot(y, p_line, 'b', linewidth=2) # plot pressure profile in blue
        plt.xlabel('$y$')
        plt.savefig(filename+'_p.png') # save the plot
        plt.clf()

        # plot temperate profile on outlet
        points = [(xmax, y_) for y_ in y] # generate the set of points in the outlet
        t_line = np.array([c(point) for point in points]) # evaluate the value of temperature on generated points
        plt.plot(y, t_line, 'r', linewidth=2) # plot temperature profile in red
        plt.xlabel('$y$')
        plt.savefig(filename+'_t.png') # save the plot
        plt.clf()

        # compute and print the weighted average value of pressure (inlet) and temperature (outlet)
        p_average = np.average(p_line)
        c_average = np.average(t_line)

        pressure_drop = 0 - p_average
        heat = (xmax - xmin) * (t_in - c_average)
        print(pressure_drop, heat)

        # # Save final solutions for next solving
        # u_0     = u_n
        # n_call += 1

    except Exception as exc:
        print(exc)
        return 0.0, 0.0, False

    # return drag, lift, True
    return pressure_drop, heat, True
