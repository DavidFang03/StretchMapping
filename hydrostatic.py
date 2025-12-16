import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constante Gravitationnelle (SI)
G = 6.67430e-11


def get_tillotson_derivatives(rho, kwargs):
    """
    Return P, dP/drho (p_prime) et d2P/drho2 (p_second) according to Tillotson in condensed state.
    """
    # Extraction des paramètres
    rho0 = kwargs["rho0"]
    E0 = kwargs["E0"]
    a = kwargs["a"]
    b = kwargs["b"]
    A = kwargs["A"]
    B = kwargs["B"]
    u_int = kwargs["u_int"]  # constant ?

    # Variables intermédiaires
    eta = rho / rho0
    chi = eta - 1.0

    if eta <= 0:
        return 0, 0, 0

    # Terme recurrent X = u / (E0 * eta^2)
    term_X = u_int / (E0 * eta**2)
    denom_1plusX = 1.0 + term_X

    # --- Pression (pour info ou debug) ---
    # P = [a + b/(1+X)] * rho * u + A*chi + B*chi^2
    P = (a + b / denom_1plusX) * rho * u_int + A * chi + B * chi**2

    # --- Dérivée première : p' ---
    # Formule : [a + b/(1+X)] * u + (2 b u^2)/(E0 eta^2 (1+X)^2) + (A + 2B chi)/rho0
    term_prime_1 = (a + b / denom_1plusX) * u_int
    term_prime_2 = (2 * b * u_int**2) / (E0 * eta**2 * denom_1plusX**2)
    term_prime_3 = (A + 2 * B * chi) / rho0

    p_prime_val = term_prime_1 + term_prime_2 + term_prime_3

    # --- Dérivée seconde : p'' ---
    # Formule : (1/rho0) * [ 8b u^3 / (E0^2 eta^5 (1+X)^3) - 2b u^2 / (E0 eta^3 (1+X)^2) + 2B/rho0 ]

    num_A = 8 * b * u_int**3
    den_A = (E0**2) * (eta**5) * (denom_1plusX**3)

    # Terme B = 2 b u^2 ...
    num_B = 2 * b * u_int**2
    den_B = E0 * (eta**3) * (denom_1plusX**2)

    term_sec_1 = num_A / den_A
    term_sec_2 = num_B / den_B
    term_sec_3 = 2 * B / rho0

    # Facteur global 1/rho0 (car d/drho = (1/rho0) d/deta)
    p_second_val = (1.0 / rho0) * (term_sec_1 - term_sec_2 + term_sec_3)

    return p_prime_val, p_second_val, P


def hydrostatic_ode(r, vec, kwargs):
    """
    Système d'équations différentielles :
    d(mu)/dr = nu
    d(nu)/dr = (nu/mu - 2/r)*nu - (p''/p')*nu^2 - (4*pi*G*mu^2)/p'
    """
    mu, nu = vec

    # Protection contre les valeurs physiques aberrantes
    if mu <= 0:
        return [nu, 0]

    p_prime_val, p_second_val, _ = get_tillotson_derivatives(mu, kwargs)

    # Gestion de la singularité en r=0
    # Si r est très petit, le terme 2/r domine.
    # Physiquement nu (drho/dr) tend vers 0 en r=0.
    if r < 1e-9:
        geom_term = 0  # Limite de (nu/mu - 2/r)*nu quand r->0 est 0
    else:
        geom_term = (nu / mu - 2.0 / r) * nu

    # Calcul de nu_prime (d2rho/dr2)
    # Eviter division par zéro si p' est nul (peu probable pour un solide comprimé)
    if p_prime_val == 0:
        dnu_dr = 0
    else:
        hydro_term = (4.0 * np.pi * G * mu**2) / p_prime_val
        thermo_term = (p_second_val / p_prime_val) * nu**2

        dnu_dr = geom_term - thermo_term - hydro_term

    return [nu, dnu_dr]


def surface_event(r, vec, kwargs):
    """Arrête l'intégration lorsque la densité tombe à une valeur plancher (ex: 1 kg/m3)"""
    mu, nu = vec
    return mu - 1.0  # On cherche le zéro de (rho - 1)


def solve_hydrostatic(kwargs):
    """Résout l'équilibre hydrostatique"""

    surface_event.terminal = True
    surface_event.direction = -1

    # Arbitrary density at the center
    mu_initial = kwargs.get("rho_center", kwargs["rho0"] * 1.5)
    nu_initial = 0.0

    vec_initial = [mu_initial, nu_initial]

    rmin = 1e-4  # Avoid r=0
    rmax = 1e8  # Large enough ?

    sol = solve_ivp(
        hydrostatic_ode,
        [rmin, rmax],
        vec_initial,
        args=(kwargs,),  # Tuple d'un seul élément
        events=surface_event,
        dense_output=True,
        rtol=1e-8,  # Tolérance fine pour la précision
        atol=1e-8,
        method="RK45",
    )

    if sol.t_events[0].size > 0:
        R_surface = sol.t_events[0][0]
    else:
        R_surface = sol.t[-1]
        print("Rmax has not been reacher")

    num_points = 200
    R_discrete = np.linspace(sol.t[0], R_surface, num_points)
    Rho_discrete = sol.sol(R_discrete)[0, :]

    return R_discrete, Rho_discrete


if __name__ == "__main__":
    # Paramètres pour le Granite (Source: Melosh 1989 / Benz code)
    # Unités SI converties depuis CGS si nécessaire
    kwargs_tillotson = {
        "rho0": 2700.0,  # kg/m^3
        "E0": 1.6e7,  # J/kg (Spécifique energy of sublimation approx)
        "a": 0.5,
        "b": 1.3,
        "A": 3.5e10,  # Pa (Bulk modulus A)
        "B": 1.8e10,  # Pa (Non-linear modulus B)
        "alpha": 5.0,  # Paramètres alpha/beta (pas utilisés dans la dérivée simple ici mais souvent dans EoS complète)
        "beta": 5.0,
        "u_int": 1e5,  # Energie interne initiale (J/kg) - "Froid"
        "rho_center": 4000.0,  # On force une densité centrale > rho0 pour voir le profil
    }

    print(
        f"Calcul pour un cœur de Granite avec rho_center = {kwargs_tillotson['rho_center']} kg/m3..."
    )

    R, Rho = solve_hydrostatic(kwargs_tillotson)

    M_total = np.trapezoid(4 * np.pi * R**2 * Rho, R)
    R_km = R[-1] / 1000.0

    print(f"Rayon final : {R_km:.2f} km")
    print(f"Masse totale : {M_total:.2e} kg")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(R / 1000.0, Rho, label="Density (Granite)", color="firebrick", linewidth=2)
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Density (kg/m³)")
    ax.set_title(
        f"Density profile (condensed Tillotson)\nR_final={R_km:.1f} km, M={M_total:.2e} kg"
    )
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    plt.show()
