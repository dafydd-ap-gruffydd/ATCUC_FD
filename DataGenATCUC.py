import firedrake as fd
import numpy as np
import csv

# Mesh
mesh = fd.PeriodicRectangleMesh(500, 500, 1.0, 1.0, direction="both")
V0 = fd.FunctionSpace(mesh, "DG", 0)
chi = fd.Function(V0)
x, y = fd.SpatialCoordinate(mesh)

# Function space for displacement
V = fd.VectorFunctionSpace(mesh, "Lagrange", 1)

# Nullspace: constant translations (rigid body modes on periodic mesh)
nullspace = fd.VectorSpaceBasis(constant=True)

nu = 0.3

# ─── Helper functions (defined once, outside the loop) ───

def epsilon(u):
    return fd.sym(fd.grad(u))

def macro_strain_2d(i):
    Eps_voigt = np.zeros(3)
    Eps_voigt[i] = 1.0
    return fd.Constant(((Eps_voigt[0], Eps_voigt[2] / 2.0),
                         (Eps_voigt[2] / 2.0, Eps_voigt[1])))

def sigma(u, E_field, lambda_, mu):
    Id = fd.Identity(2)
    return lambda_ * fd.tr(epsilon(u)) * Id + 2 * mu * epsilon(u)

def sigma_with_macro(u, Eps, E_field, lambda_, mu):
    Id = fd.Identity(2)
    eps_total = epsilon(u) + Eps
    return lambda_ * fd.tr(eps_total) * Id + 2 * mu * eps_total

def stress_to_voigt(s):
    return fd.as_vector([s[0, 0], s[1, 1], s[0, 1]])

# ─── Parameter sweep ───

r_values = np.arange(0.154, 0.402, 0.002) #last run ended at 0.152

for r_outer in r_values:
    results = []
    t_values = np.arange(0.004, 1.0, 0.002)
    seen_auxetic = False  # Track whether we've seen negative nu yet

    for t in t_values:
        r_inner = r_outer - t
        r_mid = r_inner + t / 2

        centres = [
            (0.25, 0.75), (0.75, 0.75), (0.25, 0.25), (0.75, 0.25)
        ]

        circles = fd.Constant(0.0)
        for xc, yc in centres:
            r = fd.sqrt((x - xc)**2 + (y - yc)**2)
            circle_i = fd.conditional(fd.And(r < r_outer, r > r_inner), 1, 0)
            circles = fd.max_value(circles, circle_i)

        uppermidh   = fd.conditional(fd.And(abs(y - (0.75 - r_mid)) < t/2, fd.And(x > 0.25, x < 0.75)), 1, 0)
        lowermidh   = fd.conditional(fd.And(abs(y - (0.25 + r_mid)) < t/2, fd.And(x > 0.25, x < 0.75)), 1, 0)
        upperlefth  = fd.conditional(fd.And(abs(y - (0.75 + r_mid)) < t/2, fd.And(x > 0.0,  x < 0.25)), 1, 0)
        lowerlefth  = fd.conditional(fd.And(abs(y - (0.25 - r_mid)) < t/2, fd.And(x > 0.0,  x < 0.25)), 1, 0)
        upperrighth = fd.conditional(fd.And(abs(y - (0.75 + r_mid)) < t/2, fd.And(x > 0.75, x < 1.0)),  1, 0)
        lowerrighth = fd.conditional(fd.And(abs(y - (0.25 - r_mid)) < t/2, fd.And(x > 0.75, x < 1.0)),  1, 0)

        horizontallines = uppermidh + lowermidh + upperlefth + lowerlefth + upperrighth + lowerrighth

        midleftv    = fd.conditional(fd.And(abs(x - (0.25 - r_mid)) < t/2, fd.And(y > 0.25, y < 0.75)), 1, 0)
        midrightv   = fd.conditional(fd.And(abs(x - (0.75 + r_mid)) < t/2, fd.And(y > 0.25, y < 0.75)), 1, 0)
        upperleftv  = fd.conditional(fd.And(abs(x - (0.25 + r_mid)) < t/2, fd.And(y > 0.75, y < 1.0)),  1, 0)
        upperrightv = fd.conditional(fd.And(abs(x - (0.75 - r_mid)) < t/2, fd.And(y > 0.75, y < 1.0)),  1, 0)
        lowerleftv  = fd.conditional(fd.And(abs(x - (0.25 + r_mid)) < t/2, fd.And(y > 0.0,  y < 0.25)), 1, 0)
        lowerrightv = fd.conditional(fd.And(abs(x - (0.75 - r_mid)) < t/2, fd.And(y > 0.0,  y < 0.25)), 1, 0)

        verticallines = midleftv + midrightv + upperleftv + upperrightv + lowerleftv + lowerrightv

        cell = fd.max_value(circles, fd.max_value(horizontallines, verticallines))
        chi.interpolate(fd.conditional(cell > 0, 1, 0))

        E_solid = 210e9
        E_void  = 1e-9
        E = fd.Function(V0)
        E.interpolate(E_void + chi * (E_solid - E_void))
        lambda_field = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu_field = E / (2.0 * (1.0 + nu))

        sigmacro = np.zeros((3, 3))

        for j in range(3):
            Eps = macro_strain_2d(j)

            u = fd.TrialFunction(V)
            v = fd.TestFunction(V)

            a = fd.inner(sigma(u, E, lambda_field, mu_field), epsilon(v)) * fd.dx

            Id = fd.Identity(2)
            sigma_macro = lambda_field * fd.tr(Eps) * Id + 2 * mu_field * Eps
            L = -fd.inner(sigma_macro, epsilon(v)) * fd.dx

            uh = fd.Function(V)
            fd.solve(a == L, uh,
                     solver_parameters={
                         "ksp_type": "preonly",
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"
                     },
                     nullspace=nullspace)
            nullspace.orthogonalize(uh)

            vol = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
            stress_field = sigma_with_macro(uh, Eps, E, lambda_field, mu_field)
            stress_voigt = stress_to_voigt(stress_field)

            for k in range(3):
                sigmacro[j, k] = fd.assemble(stress_voigt[k] * fd.dx) / vol

        C11 = sigmacro[0, 0]
        C12 = sigmacro[0, 1]
        C22 = sigmacro[1, 1]
        C66 = sigmacro[2, 2]

        if abs(C22) > 1e-10:
            E_eff_x = (C11 * (1 + nu) * (1 - 2 * nu)) / (1 - nu)
            E_eff_y = (C22 * (1 + nu) * (1 - 2 * nu)) / (1 - nu)
            nu_eff_xy = C12 / C22
            nu_eff_yx = C12 / C11
        else:
            E_eff_x = 0.0
            E_eff_y = 0.0
            nu_eff_xy = 0.0
            nu_eff_yx = 0.0

        results.append({
            'r_outer': r_outer,
            't': t,
            'C11': C11, 'C12': C12, 'C22': C22, 'C66': C66,
            'E_eff_x': E_eff_x, 'E_eff_y': E_eff_y,
            'nu_eff_xy': nu_eff_xy, 'nu_eff_yx': nu_eff_yx
        })

        # Only break once we've confirmed auxetic behaviour existed and then lost it
        if nu_eff_xy < 0:
            seen_auxetic = True
        elif seen_auxetic and nu_eff_xy > 0:
            print(f"R={r_outer:.3f} — auxetic behaviour lost at t={t:.4f}")
            break

        # Skip this R entirely if no auxetic behaviour by t=0.1
        if not seen_auxetic and t >= 0.1:
            print(f"R={r_outer:.3f} — no auxetic behaviour by t=0.1, skipping")
            results = []
            break

    if results:
        with open(f"Data(R={r_outer:.3f}).csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"R={r_outer:.3f} — wrote {len(results)} rows")