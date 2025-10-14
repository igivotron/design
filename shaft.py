import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument('-T', type=float, help="Couple en Nm", default=36500)
parser.add_argument('-s', type=str, help="Yield limit du shaft", default=250e6)
parser.add_argument('-p', type=int, help="1 afficher, 0 pas afficher", default=1)
parser.add_argument('-m', type=int, help="1 pour comparer les méthodes de minimisation, 0 non", default=0)
parser = parser.parse_args()

T = parser.T
sigma_yield = float(parser.s)
plot = parser.p
methods = parser.m

# Constants
alpha = 20 # degree - angle rouage
Rd = 0.5 # m - rayon de la roue dentée
Rp = 0.5 # m - rayon de la poulie

# Forces
Fp = -T / Rp
Fu = T / Rd
Fr = -Fu * np.tan(np.radians(alpha))

def compute_reactions(Fp, Fr, Fu, L1, L2, L3):

    FCy = L1/L2 * Fp + (1 + L3/L2) * Fr
    FBy = Fr - Fp - FCy

    FCz = (1 + L3/L2) * Fu
    FBz = Fu - FCz

    FCz = - FCz
    FBz = - FBz
    return FBy, FBz, FCy, FCz

def FABy(x, Fp):
    return Fp
def FBCy(x, FBy, Fp):
    return FBy + Fp
def FCDy(x, FCy, FBy, Fp):
    return FCy + FBy + Fp

def FDinfy(x, FCy, FBy, Fp, Fr):
    return FCy + FBy + Fp - Fr

def MABy(x, Fp):
    return x*Fp
def MBCy(x, FBy, Fp, L1):
    return Fp*L1 + (x-L1) * (FBy + Fp)
def MCDy(x, FCy, FBy, Fp, L1, L2):
    return L1 * Fp + L2 * (FBy + Fp) + (x-L1 -L2) * (FCy + FBy + Fp)

def FABz(x, FBz):
    return 0
def FBCz(x, FBz):
    return FBz
def FCDz(x, FCz, FBz):
    return FCz + FBz
def FDinfz(x, FCz, FBz, Fu):
    return FCz + FBz + Fu

def MABz(x):
    return 0

def MBCz(x, FBz, L1):
    return FBz * (x - L1)

def MCDz(x, L1, L2, FBz, FCz):
    return L2*FBz + (x - L1 - L2) * (FCz + FBz)

def MMax(My, Mz):
    return np.max(np.sqrt(My**2 + Mz**2))

def get_d(My, Mz, T, sigma_yield):
    M = MMax(My, Mz)
    # BORDEL CA PRENDS LA TORSION EN COMPTE
    return ((32*M/np.pi)**2 + 3*(16*T/np.pi)**2)**(1/6) / (sigma_yield**(1/3))

def torsion_stress(T, d):
    return 16*T/(np.pi*d**3)
    

def compute_forces(L1, L2, L3, Fp, Fr, Fu, FBy, FBz, FCy, FCz):
    x = np.linspace(0, L1 + L2 + L3, 1000)
    Fy = np.piecewise(x, [x <= L1, (x > L1) & (x <= L1 + L2), (x > L1 + L2) & (x <= L1 + L2 + L3), x >= (L1 + L2 + L3)], [lambda x: FABy(x, Fp), lambda x: FBCy(x, FBy, Fp), lambda x: FCDy(x, FCy, FBy, Fp), lambda x: FDinfy(x, FCy, FBy, Fp, Fr)])
    My = np.piecewise(x, [x <= L1, (x > L1) & (x <= L1 + L2), x > L1 + L2], [lambda x: MABy(x, Fp), lambda x: MBCy(x, FBy, Fp, L1), lambda x: MCDy(x, FCy, FBy, Fp, L1, L2)])
    Fz = np.piecewise(x, [x <= L1, (x > L1) & (x <= L1 + L2), (x > L1 + L2) & (x <= L1 + L2 + L3), x >= (L1 + L2 + L3)], [lambda x: FABz(x, FBz), lambda x: FBCz(x, FBz), lambda x: FCDz(x, FCz, FBz), lambda x: FDinfz(x, FCz, FBz, Fu)])
    Mz = np.piecewise(x, [x <= L1, (x > L1) & (x <= L1 + L2), x > L1 + L2], [lambda x: MABz(x), lambda x: MBCz(x, FBz, L1), lambda x: MCDz(x, L1, L2, FBz, FCz)])
    return Fy, My, Fz, Mz, x


def optimize_d(L, Fp, Fr, Fu, sigma_yield):
    L1, L2, L3 = L
    FBy, FBz, FCy, FCz = compute_reactions(Fp, Fr, Fu, L1, L2, L3)
    Fy, My, Fz, Mz, _ = compute_forces(L1, L2, L3, Fp, Fr, Fu, FBy, FBz, FCy, FCz)
    d = get_d(My, Mz, T, sigma_yield)
    return d

### Optimization
bounds = [(0.1, 1), (0.1, 1), (0.1, 1)]
if not methods:
    res = sp.optimize.minimize(lambda L: optimize_d(L, Fp, Fr, Fu, sigma_yield), x0=[0.6, 0.5, 0.4], bounds=bounds, method='Powell')
    print("Optimal lengths (L1, L2, L3):", res.x)
    print("Minimum diameter d (m):", res.fun)
    L1, L2,  L3 = res.x

if methods:
    m = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
    d = np.zeros(len(m))
    for i in range(len(m)):
        res = sp.optimize.minimize(lambda L: optimize_d(L, Fp, Fr, Fu, sigma_yield), x0=[0.6, 0.5, 0.4], bounds=bounds, method=m[i])
        d[i] = res.fun
        print(f"Method: {m[i]}")
        print("  Optimal lengths (L1, L2, L3):", res.x)
        print("  Minimum diameter d (m):", res.fun)
    
    L1, L2,  L3 = res.x
    
    plt.figure(figsize=(10, 6))
    plt.bar(m, d)
    plt.ylabel('Minimum Diameter d (m)')
    plt.title('Comparison of Optimization Methods')
    plt.grid(axis='y')
    plt.show()

print("Optimal lengths (L1, L2, L3):", L1, L2, L3)

if plot:
    # Y direction
    FBy, FBz, FCy, FCz = compute_reactions(Fp, Fr, Fu, L1, L2, L3)
    Fy, My, Fz, Mz, x = compute_forces(L1, L2, L3, Fp, Fr, Fu, FBy, FBz, FCy, FCz)
    d = get_d(My, Mz, T, sigma_yield)
    print("d (m):", d)

    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=1.5)
    plt.gca().invert_yaxis()
    plt.plot(x, Fy, label='Shear Force (N)')
    plt.plot(x, My, label='Bending Moment (N.m)')
    plt.fill_between(x, 0, Fy, color='lightblue', alpha=0.5)
    plt.fill_between(x, 0, My, color='orange', alpha=0.5)

    # draw arrows for forces
    plt.arrow(0, 0, 0, Fp, head_width=0.02, head_length=0.05, fc='red', ec='red')
    plt.arrow(L1, 0, 0, FBy, head_width=0.02, head_length=0.05, fc='red', ec='red')
    plt.arrow(L1 + L2, 0, 0, FCy, head_width=0.02, head_length=0.05, fc='red', ec='red')
    plt.arrow(L1 + L2 + L3, 0, 0, -Fr, head_width=0.02, head_length=0.05, fc='red', ec='red')

    plt.title('Shear Force and Bending Moment Diagram')
    plt.xlabel('Position along the beam (m)')
    plt.ylabel('Force (N) / Moment (N.m)')
    plt.legend()
    plt.grid()

    # Z direction
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=1.5)
    plt.gca().invert_yaxis()
    plt.plot(x, Fz, label='Shear Force (N)')
    plt.plot(x, Mz, label='Bending Moment (N.m)')
    plt.fill_between(x, 0, Fz, color='lightblue', alpha=0.5)
    plt.fill_between(x, 0, Mz, color='orange', alpha=0.5)

    plt.arrow(L1, 0, 0, FBz, head_width=0.02, head_length=0.05, fc='red', ec='red')
    plt.arrow(L1 + L2, 0, 0, FCz, head_width=0.02, head_length=0.05, fc='red', ec='red')
    plt.arrow(L1 + L2 + L3, 0, 0, Fu, head_width=0.02, head_length=0.05, fc='red', ec='red')

    plt.title('Shear Force and Bending Moment Diagram (Z direction)')
    plt.xlabel('Position along the beam (m)')
    plt.ylabel('Force (N) / Moment (N.m)')
    plt.legend()
    plt.grid()

    # Torsion stress
    plt.figure(figsize=(10, 6))
    torsion_stress_values = torsion_stress(T, d)*np.ones_like(x)
    plt.plot(x, torsion_stress_values, label='Torsion Stress (Pa)', color='green')
    plt.title('Torsion Stress along the Shaft')
    plt.xlabel('Position along the beam (m)')
    plt.ylabel('Stress (Pa)')
    plt.legend()
    plt.grid()

    plt.show()