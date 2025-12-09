import numpy as np


def integrate_profile(rhoprofile, rmin, r, dim=1, ngrid=8192):
    dr = (r - rmin) / ngrid
    mass = 0.0
    r_prev = rmin
    f_prev = rhoprofile(r_prev) * dS(r_prev, dim)
    for i in range(1, ngrid):
        r_curr = rmin + i * dr
        f_curr = rhoprofile(r_curr) * dS(r_curr, dim)
        mass += 0.5 * (f_prev + f_curr) * dr
        f_prev = f_curr
    return mass


def dS(r, dim=1):
    match dim:
        case 1:
            return 1
        case 2:
            return 2 * np.pi * r
        case 3:
            return 4 * np.pi * r**2


def integrate_target(target, dim=3):
    dr = np.diff(target[0])
    return np.sum(target[1, :-1] * dr * dS(target[0, :-1], dim))


# Constantes fondamentales
h = 6.62607015e-34  # Constante de Planck (J s)
c = 2.99792458e8  # Vitesse de la lumière (m/s)
m_e = 9.1093837015e-31  # Masse de l'électron (kg)
m_p = 1.67262192369e-27  # Masse du proton (kg)
# Poids moléculaire moyen par électron (μ=2.0 pour les naines blanches typiques composées de C/O)
mu = 2.0
G = 6.67e-11

Msol = 1.989e30
Rsol = 6.957e8
rhoadim = Msol / (Rsol**3)


def get_rho(phi, y0, rho0):
    return (
        rho0
        * np.pow(y0, 3)
        / np.pow(y0**2 - 1, 3.0 / 2.0)
        * np.pow(phi**2 - 1 / y0**2, 3.0 / 2.0)
    )


def chandrasekhar_ode(eta, y, y0):
    """
    y = [u, v] = [Phi, Phi']
    y0 est le paramètre du modèle
    """
    u, v = y

    # 1. Calcul du terme RHS (partie non singulière)
    arg = u**2 - (1 / y0) ** 2

    # Si arg < 0, la densité/pression est nulle, donc l'évolution s'arrête
    if arg < 0:
        rhs_v = 0.0
    else:
        rhs_v = -((arg) ** (3 / 2))

    # 2. Gestion du terme singulier (2/eta * v)
    eta_min_threshold = 1e-6

    if eta < eta_min_threshold:
        # Près du centre, le terme singulier (2/eta * v) tend vers 0
        # car v=Phi' tend vers 0 plus rapidement que 2/eta ne diverge.
        # On utilise l'approximation de Taylor (Phi'(eta) ~ -A*eta/3),
        # ce qui annule le terme: (2/eta) * (-A*eta/3) = -2A/3 (constante)
        # Mais puisqu'on démarre l'IVP à eta_min, on peut le poser à 0
        # car le développement en série est déjà utilisé pour y_initial.
        term_singulier = 0.0
    else:
        # Loin du centre, on utilise la formule standard
        term_singulier = (2 / eta) * v

    # 3. Retourner le système (u' = v, v' = RHS - terme_singulier)
    return [v, rhs_v - term_singulier]


# --- 2. Définir l'événement d'arrêt (la surface)
def surface_event(eta, y, y0):
    """Arrête l'intégration lorsque Phi = 1/y0 (la densité s'annule)"""
    u, v = y
    return u - (1 / y0)


def solve_Chandrasekhar(y_0):
    from scipy.integrate import solve_ivp

    # L'événement doit s'arrêter lorsque le signe change (is_terminal=True)
    surface_event.terminal = True
    surface_event.direction = -1  # On s'attend à ce que Phi diminue

    eta_min = 1e-6
    # Approximation de Taylor (Phi(eta) ~ 1 - A*eta^2/6, Phi'(eta) ~ -A*eta/3)
    A = (1 - (1 / y_0) ** 2) ** (3 / 2)
    u_initial = 1.0 - (A / 6) * eta_min**2
    v_initial = -(A / 3) * eta_min

    y_initial = [u_initial, v_initial]

    # --- 4. Résoudre l'IVP
    eta_span = [eta_min, 10.0]  # Intégrer jusqu'à un rayon max suffisant
    sol = solve_ivp(
        chandrasekhar_ode,
        eta_span,
        y_initial,
        args=(y_0,),
        events=surface_event,
        dense_output=True,
        method="RK45",
    )

    eta_s = sol.t_events[0][0]
    num_points = 100
    eta_discret = np.linspace(sol.t[0], eta_s, num_points)
    solution_discrète = sol.sol(eta_discret)[0, :]

    C1 = (8 * np.pi * m_e**3 * c**3 * m_p * mu) / (3 * h**3)

    C2 = (np.pi * m_e**4 * c**5) / (3 * h**3)
    alpha = np.sqrt(2 * C2 / (np.pi * G)) / (C1 * y_0)
    rho_0 = C1 * np.pow(y_0 - 1, 1.5)

    R = alpha * eta_discret / Rsol
    RHO = get_rho(solution_discrète, y_0, rho_0) / rhoadim

    print(R.shape, RHO.shape)

    return R, RHO


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ###########################################
    y0_value = 1.5  # Exemple : paramètre de relativité
    ###########################################

    R, RHO = solve_Chandrasekhar(y0_value)

    fig, ax = plt.subplots()
    ax.plot(R, RHO)

    plt.show()

# sol.t_events[0] donne l'emplacement du rayon de surface eta_s
# sol.y[0] est la solution pour Phi, sol.y[1] est la solution pour Phi'
