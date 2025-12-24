import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import stretchmap_utilities as su

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
    G = kwargs.get("G")

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
    """Stop integration when P = 0"""
    mu, nu = vec
    pressure = get_tillotson_pressure_sound(mu, kwargs["u_int"], kwargs)[0][-1]
    return pressure


def solve_hydrostatic(kwargs, unit):
    """
    DEPRECATED
    Résout l'équilibre hydrostatique"""

    surface_event.terminal = True
    surface_event.direction = -1

    # Arbitrary density at the center
    mu_initial = kwargs.get("rho_center", kwargs["rho0"] * 1.5)
    nu_initial = 0.0

    vec_initial = [mu_initial, nu_initial]

    rmin = 1e-4  # Avoid r=0
    rmax = 1e8  # Should be large enough

    length = unit.to("metre")
    time = unit.to("second")
    mass = unit.to("kilogram")
    unitG = length**3 / mass / time**2
    G = 6.67e-11 / unitG
    print("G", G)
    kwargstoivp = kwargs.copy()
    kwargstoivp["G"] = G

    sol = solve_ivp(
        hydrostatic_ode,
        [rmin, rmax],
        vec_initial,
        args=(kwargstoivp,),
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
        print("Rmax has not been reached !!!")

    num_points = 200
    R_discrete = np.linspace(sol.t[0], R_surface, num_points)
    Rho_discrete = sol.sol(R_discrete)[0, :]

    mask_unphysical = (
        get_tillotson_pressure_sound(Rho_discrete, kwargs["u_int"], kwargs)[0] <= 0
    )

    Rho_discrete[mask_unphysical] = 1e-6

    return R_discrete, Rho_discrete


def solve_hydrostatic_tillotson(tillotson_params, rho_center, u_int, unit):
    """
    Returns tabx, tabrho by solving Tillotson+Hydrostatic equilibrium

    Input: everything must be in the same unit. Unit still has to be precised beaucause of G.
    """

    surface_event.terminal = True
    surface_event.direction = -1

    # Arbitrary density at the center
    mu_initial = rho_center
    nu_initial = 0.0

    vec_initial = [mu_initial, nu_initial]

    rmin = 1e-4  # Avoid r=0
    rmax = 1e8  # Should be large enough

    length = unit.to("metre")
    time = unit.to("second")
    mass = unit.to("kilogram")
    unitG = length**3 / mass / time**2
    G = 6.67e-11 / unitG
    print("G", G)
    kwargstoivp = tillotson_params.copy()
    kwargstoivp["G"] = G
    kwargstoivp["u_int"] = u_int

    sol = solve_ivp(
        hydrostatic_ode,
        [rmin, rmax],
        vec_initial,
        args=(kwargstoivp,),
        events=surface_event,
        dense_output=True,
        rtol=1e-8,  # TODO what is that Tolérance fine pour la précision
        atol=1e-8,  # TODO what is that
        method="RK45",
    )

    if sol.t_events[0].size > 0:
        R_surface = sol.t_events[0][0]
    else:
        R_surface = sol.t[-1]
        print("Rmax has not been reached !!!")

    num_points = 200
    R_discrete = np.linspace(sol.t[0], R_surface, num_points)
    Rho_discrete = sol.sol(R_discrete)[0, :]

    mask_unphysical = (
        get_tillotson_pressure_sound(Rho_discrete, u_int, tillotson_params)[0] <= 0
    )

    Rho_discrete[mask_unphysical] = 1e-6

    return R_discrete, Rho_discrete


import numpy as np


def get_tillotson_pressure_sound(rho, u, p):
    """
    Returns (P,cs) Tillotson.

    Args:
        rho: Density (float or array)
        u:   Internal energy (float or array)
        p:   Tillotson parameters {'rho0', 'a', 'b', 'A', 'B', 'E0', 'alpha', 'beta', 'u_iv', 'u_cv'}
    """
    # 1. Data structure
    rho = np.atleast_1d(rho)
    u = np.atleast_1d(u)
    n_points = len(rho)
    if len(u) == 1:
        u = [u[0] for _ in range(n_points)]

    P_out = np.zeros(n_points)
    cs_out = np.zeros(n_points)

    # 2. Parameters
    rho0 = p["rho0"]
    E0 = p["E0"]
    a, b = p["a"], p["b"]
    A, B = p["A"], p["B"]
    alpha, beta = p["alpha"], p["beta"]
    u_iv = p.get("u_iv", 0.0)
    u_cv = p.get("u_cv", 1e30)
    delta_u = u_cv - u_iv

    # 3. Loop
    for i in range(n_points):
        r = rho[i]  # densité locale
        eng = u[i]  # énergie locale

        # Variables communes
        eta = r / rho0
        mu = eta - 1.0
        omega = eng / (E0 * (eta**2))
        denom = 1.0 + omega

        # Initialisation des variables temporaires
        P_val, dPr, dPu = 0.0, 0.0, 0.0

        # --- DÉTECTION DE LA RÉGION ---
        is_cold = False
        is_hot = False

        # Masque logique simple
        if (r >= rho0) or (eng < u_iv):
            # Région Froide / Compressée
            is_cold = True
            regime = "COLD"
        elif (r < rho0) and (eng > u_cv):
            # Région Chaude / Vaporisée
            is_hot = True
            regime = "HOT"
        else:
            # Région Intermédiaire (nécessite les deux calculs)
            is_cold = True
            is_hot = True
            regime = "INTER"

        # --- CALCULS PHYSIQUES ---

        # Stockage temporaire pour l'interpolation
        Pc, dPrc, dPuc = 0.0, 0.0, 0.0
        Ph, dPrh, dPuh = 0.0, 0.0, 0.0

        # A. Formules "FROIDES" (Cold)
        if is_cold:
            # Pression
            Pc = (a + b / denom) * r * eng + A * mu + B * (mu**2)
            if Pc < 0:
                Pc = 0

            # dP/du
            dPuc = r * (a + b / (denom**2))

            # dP/drho
            term1 = eng * (a + b / denom)
            # d(omega)/drho = -2*omega/rho.  d(1/(1+w))/drho = -1/(1+w)^2 * dw/dr
            term2 = (2.0 * b * eng * omega) / (denom**2)
            term3 = A / rho0
            term4 = (2.0 * B * mu) / rho0
            dPrc = term1 + term2 + term3 + term4

        # B. Formules "CHAUDES" (Hot)
        if is_hot:
            X = (1.0 / eta) - 1.0  # (rho0/rho) - 1
            exp_beta = np.exp(-beta * X)
            exp_alpha = np.exp(-alpha * (X**2))

            bracket = (b * r * eng) / denom + A * mu * exp_beta

            # Pression
            Ph = a * r * eng + bracket * exp_alpha

            # dP/du
            dPuh = a * r + (b * r / (denom**2)) * exp_alpha

            # dP/drho
            dX_drho = -1.0 / (rho0 * eta**2)  # dérivée de X par rapport à rho
            d_exp_alpha = exp_alpha * (-2.0 * alpha * X) * dX_drho
            d_exp_beta = exp_beta * (-beta) * dX_drho

            # Dérivée du terme entre crochets
            # d(b*rho*u / (1+w)) / drho
            d_term1 = (b * eng * denom - (b * r * eng) * (-2 * omega / r)) / (denom**2)
            # d(A*mu*exp_beta) / drho
            d_term2 = A * ((1.0 / rho0) * exp_beta + mu * d_exp_beta)

            d_bracket = d_term1 + d_term2

            dPrh = a * eng + d_bracket * exp_alpha + bracket * d_exp_alpha

        # --- COMBINAISON DES RESULTATS ---

        if regime == "COLD":
            P_val = Pc
            dPr = dPrc
            dPu = dPuc

        elif regime == "HOT":
            P_val = Ph
            dPr = dPrh
            dPu = dPuh

        elif regime == "INTER":
            # Interpolation linéaire
            x = (eng - u_iv) / delta_u  # Poids du chaud
            y = (u_cv - eng) / delta_u  # Poids du froid

            P_val = x * Ph + y * Pc

            # Dérivées interpolées
            dPr = x * dPrh + y * dPrc
            # Pour dP/du, on applique la règle du produit sur les poids x(u) et y(u)
            # dx/du = 1/delta, dy/du = -1/delta
            dPu = (Ph - Pc) / delta_u + (x * dPuh + y * dPuc)

        # 3. Calcul final Vitesse du son
        P_out[i] = P_val

        # c^2 = dP/drho + (P / rho^2) * dP/du
        c2 = dPr + (P_val / (r**2)) * dPu

        if c2 > 0:
            cs_out[i] = np.sqrt(c2)
        else:
            cs_out[i] = 1e-8  # Valeur plancher pour éviter plantage

    return P_out, cs_out


Rearth = 6371e3
Mearth = 5.972e24
Tearth = np.sqrt(Rearth**3 / (Mearth * 6.67e-11))


class Unitsystem:
    def __init__(self, unit_name="earth"):
        if unit_name == "earth":
            self.length = Rearth
            self.time = Tearth
            self.mass = Mearth
        pass

    def to(self, name):
        if name == "metre":
            return self.length
        elif name == "second":
            return self.time
        elif name == "kilogram":
            return self.mass


def adimension(tillotson_values, unit):
    length = unit.to("metre")
    time = unit.to("second")
    mass = unit.to("kilogram")
    density = mass / (length**3)
    energy = (length**2) / (time**2)
    pressure = mass / (length * (time**2))

    # --- Application des conversions ---

    # Groupe DENSITÉ (rho0, rho_center, etc.)
    for key in ["rho0", "rho_center", "rho_ref"]:
        if key in tillotson_values:
            tillotson_values[key] /= density

    # Groupe ÉNERGIE SPÉCIFIQUE (E0, u_iv, u_cv, u_int, u...)
    # u_iv = incipient vaporization, u_cv = complete vaporization
    for key in [
        "E0",
        "u_iv",
        "u_cv",
        "u_int",
    ]:
        if key in tillotson_values:
            tillotson_values[key] /= energy

    # Groupe PRESSION (A, B)
    for key in ["A", "B", "P_ref"]:
        if key in tillotson_values:
            tillotson_values[key] /= pressure

    # Les constantes sans dimension (a, b, alpha, beta) restent inchangées.

    return tillotson_values


if __name__ == "__main__":
    codeearth = Unitsystem()
    # Paramètres pour le Granite (Source: Melosh 1989 / Benz code)
    # Unités SI converties depuis CGS si nécessaire
    # kwargs_tillotson = {
    #     "rho0": 7.8e3,  # kg/m^3
    #     "E0": 0.095e8,  # J/kg (Spécifique energy of sublimation approx)
    #     "a": 0.5,
    #     "b": 1.5,
    #     "A": 1.279e11,  # Pa (Bulk modulus A)
    #     "B": 1.05e11,  # Pa (Non-linear modulus B)
    #     "alpha": 5.0,
    #     "beta": 5.0,
    #     "u_iv": 0.024e8,
    #     "u_cv": 0.0867e8,
    # }
    #
    material = "Granite"
    if material == "Granite":
        from balls import Tillotson_parameters_Granite

        kwargs_tillotson = Tillotson_parameters_Granite

    # kwargs_tillotson = adimension(kwargs_tillotson, codeearth)
    # print(kwargs_tillotson)

    # print(f"Fe avec rho_center = {kwargs_tillotson['rho_center']} kg/m3...")

    # tabx, tabrho = solve_hydrostatic(kwargs_tillotson, codeearth)

    # M_total = su.integrate_target([tabx, tabrho])
    # Rmax = tabx[-1]

    # eos = {"name": "tillotson", "id": f"tillotson", "values": kwargs_tillotson}
    # import stretchmap_utilities as su

    # # P_cs_func = su.get_p_and_cs_func(eos, codeearth)
    # mask = tabrho != 0
    # # P, cs = P_cs_func(tabrho[mask])
    # print("coucou")

    # use_shamrock = True
    # if use_shamrock:
    #     import shamrock
    #     import test_tillotson as tt

    #     kw_to_sham = tt.recover_tillotson_values(kwargs_tillotson)
    #     print(kw_to_sham)
    #     P, cs = [], []
    #     for rho in tabrho[mask]:

    #         p, _cs = shamrock.phys.eos.eos_Tillotson(
    #             rho=rho, u=kwargs_tillotson["u_int"], **kw_to_sham
    #         )
    #         P.append(p)
    #         cs.append(_cs)

    # print(f"Rayon final : {Rmax:.2f}")
    # print(f"Masse totale : {M_total:.2e}")

    # fig, axs = plt.subplots(3, figsize=(8, 6))
    # fig.subplots_adjust(hspace=0.5)
    # axs[0].plot(
    #     tabx, tabrho, label="Density (Granite)", color="blue", linewidth=2, marker="+"
    # )
    # axs[1].plot(tabx, cs, marker="+", color="magenta")
    # axs[2].plot(tabx, P, marker="+", color="green")

    # for ax in axs:
    #     ax.set_xlabel("Radius")
    # axs[0].set_ylabel("Density")
    # axs[1].set_ylabel("Soundspeed")
    # axs[2].set_ylabel("Pressure")
    # axs[0].set_title(
    #     f"Density profile (condensed Tillotson)\nR_final={Rmax:.1f}, M={M_total:.2e}"
    # )
    # ax.grid(True, linestyle="--", alpha=0.7)
    # ax.legend()

    # import pandas as pd

    # df = pd.DataFrame({"r": tabx, "rho_target": tabrho, "P_target": P, "cs_target": cs})
    # df.to_csv(
    #     f"rhotarget_tillotson_rhocenter_{kwargs_tillotson["rho_center"]:.0e}.csv",
    #     index=False,
    # )

    import unitsystem

    siunit = unitsystem.Unitsystem("SI")
    rho0 = kwargs_tillotson["rho0"]
    figmass, axmass = plt.subplot_mosaic(
        [["m", "profile"], ["r", "profile"]], figsize=(12, 8)
    )
    figmass.suptitle(material)
    figmass.subplots_adjust(hspace=0.5)
    rhocenter_array = np.linspace(1.1 * rho0, 7 * rho0)
    mtot_array = []
    rmax_array = []
    for i, rhocenter in enumerate(rhocenter_array):
        tabx, tabrho = solve_hydrostatic_tillotson(
            kwargs_tillotson, rhocenter, u_int=0, unit=siunit
        )
        mtot = su.integrate_target([tabx, tabrho])
        rmax = np.max(tabx[tabrho > 10])
        mtot_array.append(mtot)
        rmax_array.append(rmax)

        if i % 3 == 0:
            line = axmass["profile"].plot(tabx, tabrho, label=f"{rhocenter/rho0:.1f}")[
                0
            ]
            axmass["profile"].axvline(
                x=rmax, ls="--", alpha=0.4, color=line.get_color()
            )

    axmass["profile"].set_ylabel(r"$\rho$")
    axmass["profile"].set_title(
        r"Density profile for different $\rho_{\rm center}/\rho_0$"
    )
    axmass["m"].plot(rhocenter_array / rho0, mtot_array)
    axmass["m"].axhline(
        y=siunit.Mearth, color="blue", ls="--", alpha=0.5, label="Earth"
    )
    axmass["m"].axhline(y=siunit.Mmoon, color="black", ls="--", alpha=0.5, label="Moon")
    axmass["m"].set_ylabel(r"$M_{\rm tot}$")
    axmass["m"].set_title(r"Total mass evolution with $\rho_{\rm center}/\rho_0$")

    axmass["r"].plot(rhocenter_array / rho0, rmax_array)
    axmass["r"].axhline(
        y=siunit.Rearth, color="blue", ls="--", alpha=0.5, label="Earth"
    )
    axmass["r"].axhline(y=siunit.Rmoon, color="black", ls="--", alpha=0.5, label="Moon")
    axmass["r"].set_ylabel(r"$R_{\rm max}$")
    axmass["r"].set_title(r"Planet radius evolution with $\rho_{\rm center}/\rho_0$")

    for label, ax in axmass.items():
        if label != "profile":
            ax.set_xlabel(r"$\rho_{\rm center}/\rho_0$")
        else:
            ax.set_xlabel(r"$r$")
        ax.legend()

    plt.show()

# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./hydrostatic.py
