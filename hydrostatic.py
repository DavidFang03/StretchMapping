import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import stretchmap_utilities as su
import unitsystem

# Constante Gravitationnelle (SI)
G = 6.67430e-11

Tillotson_parameters_Fe = {
    "rho0": 7.8e3,  # kg/m^3
    "E0": 0.095e8,  # J/kg (Spécifique energy of sublimation approx)
    "a": 0.5,
    "b": 1.5,
    "A": 1.279e11,  # Pa (Bulk modulus A)
    "B": 1.05e11,  # Pa (Non-linear modulus B)
    "alpha": 5.0,
    "beta": 5.0,
    "u_iv": 0.024e8,  # TODO check this
    "u_cv": 0.0867e8,
}

Tillotson_parameters_Granite = {
    "rho0": 2.7e3,  # kg/m^3
    "E0": 1.6e7,  # J/kg (Spécifique energy of sublimation approx)
    "a": 0.5,
    "b": 1.3,
    "A": 1.8e10,  # Pa (Bulk modulus A)
    "B": 1.8e10,  # Pa (Non-linear modulus B)
    "alpha": 5.0,
    "beta": 5.0,
    "u_iv": 3.5e6,
    "u_cv": 1.8e7,
}


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
    T = kwargs["T"]  # constant ?

    # Variables intermédiaires
    u_int = get_cold_energy(rho, material="Granite", unit=kwargs["unit"])
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


def get_cold_energy(rho, material, unit):
    """
    Return cold_energy (SI)

    :param rho: densiy (SI)
    :param material: Granite, etc..
    """
    rhoa = rho * unitsystem.density(unit)
    if material == "Granite":
        # SI fit parameters that work well for Granite
        a = 3.0021e-1
        b = -9.5284e2
        c = 4.2450e5

    uc = a * rhoa * rhoa + b * rhoa + c
    return uc / unitsystem.sp_energy(unit)


def hydrostatic_ode(r, vec, kwargs_ivp):
    """
    Système d'équations différentielles :
    d(mu)/dr = nu
    d(nu)/dr = (nu/mu - 2/r)*nu - (p''/p')*nu^2 - (4*pi*G*mu^2)/p'
    """
    mu, nu = vec
    unit = kwargs_ivp.get("unit")
    length = unit.to("metre")
    time = unit.to("second")
    mass = unit.to("kilogram")
    unitG = length**3 / mass / time**2
    G = 6.67e-11 / unitG

    # Protection contre les valeurs physiques aberrantes
    if mu <= 0:
        return [nu, 0]

    p_prime_val, p_second_val, _ = get_tillotson_derivatives(mu, kwargs_ivp)

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


def solve_hydrostatic_tillotson(tillotson_params, rho_center, T, unit):
    """
    Returns tabx, tabrho by solving Tillotson+Hydrostatic equilibrium

    Input: everything must be in the same unit. Unit still has to be precised beaucause of G.
    """

    def surface_event(r, vec, kwargs):
        """Stop integration when P = 0"""
        mu, nu = vec
        pressure = get_tillotson_pressure_sound(mu, kwargs["T"], kwargs, kwargs["unit"])[0][-1]
        return pressure

    surface_event.terminal = True
    surface_event.direction = -1

    # Arbitrary density at the center
    mu_initial = rho_center
    nu_initial = 0.0

    vec_initial = [mu_initial, nu_initial]

    rmin = 1e-4  # Avoid r=0
    rmax = 1e8  # Should be large enough

    print("G", G)
    kwargstoivp = tillotson_params.copy()
    kwargstoivp["unit"] = unit
    kwargstoivp["T"] = T

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

    mask_unphysical = get_tillotson_pressure_sound(Rho_discrete, T, tillotson_params, unit)[0] <= 0

    Rho_discrete[mask_unphysical] = 1e-6

    return R_discrete, Rho_discrete


def solve_energy_cold(params, unit):
    """
    Solves cold energy curve. Return lambda function u_c(rho)

    :param rho0: Description
    :param params: Tillotson eos params
    """

    def energy_cold_ode(rho, u, params):
        return get_tillotson_pressure_sound(rho, u, params, unit)[0] / rho**2

    rho0 = params["rho0"]
    sol = solve_ivp(
        energy_cold_ode,
        [rho0, 8 * rho0],
        [0],
        args=(params,),
        dense_output=True,
        rtol=1e-8,  # TODO what is that Tolérance fine pour la précision
        atol=1e-8,  # TODO what is that
        method="RK45",
    )
    solm = solve_ivp(
        energy_cold_ode,
        [rho0, 1e-4],
        [0],
        args=(params,),
        dense_output=True,
        rtol=1e-8,  # TODO what is that Tolérance fine pour la précision
        atol=1e-8,  # TODO what is that
        method="RK45",
    )

    num_points = 200
    rhop = np.linspace(rho0, 8 * rho0, num_points)[1:]
    rhom = np.linspace(0, rho0, num_points)
    up = sol.sol(rhop)[0, :]
    um = solm.sol(rhom)[0, :]

    rhotab = np.concatenate((rhom, rhop))
    utab = np.concatenate((um, up))

    # ufunc = lambda rho: sol.sol(rho)[0, :]
    return rhotab, utab


def get_tillotson_pressure_sound(rho, T, params, unit):
    """
    Returns (P,cs) Tillotson.

    Args:
        rho: Density (float or array)
        u:   Internal energy (float or array)
        p:   Tillotson parameters {'rho0', 'a', 'b', 'A', 'B', 'E0', 'alpha', 'beta', 'u_iv', 'u_cv'}
    """
    # 1. Data structure
    rho = np.atleast_1d(rho)
    u = get_cold_energy(rho, material="Granite", unit=unit)
    n_points = len(rho)
    if len(u) == 1:
        u = [u[0] for _ in range(n_points)]

    P_out = np.zeros(n_points)
    cs_out = np.zeros(n_points)

    # 2. Parameters
    rho0 = params["rho0"]
    E0 = params["E0"]
    a, b = params["a"], params["b"]
    A, B = params["A"], params["B"]
    alpha, beta = params["alpha"], params["beta"]
    u_iv = params.get("u_iv", 0.0)
    u_cv = params.get("u_cv", 1e30)
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


def solve_Chandrasekhar(mu_e, y_0, unit):
    """
    Return in the given unit system.

    :param y_0: Input value, no dimension (=(M,R))
    :param mu_e: EoS parameter, no dimension
    :param unit: UnitSystem instance
    """
    from scipy.integrate import solve_ivp

    def surface_event(eta, y, y0):
        """Arrête l'intégration lorsque Phi = 1/y0 (la densité s'annule)"""
        u, v = y
        return u - (1 / y0)

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
    eta_discrete = np.linspace(sol.t[0], eta_s, num_points)[:-1]
    Phi_discrete = sol.sol(eta_discrete)[0, :]

    # C1 = (8 * np.pi * m_e**3 * c**3 * m_p * mu) / (3 * h**3)

    # C2 = (np.pi * m_e**4 * c**5) / (3 * h**3) = BETA
    # r_a = np.sqrt(2 * C2 / (np.pi * G)) / (C1 * y_0)
    # rho_0 = C1 * np.pow(y_0 - 1, 1.5)

    C1 = 981.0189489250643e6  # *mu_e
    RA = 0.02439045021149552 * 10 ** (8.5)  # /mu_e

    R = RA * eta_discrete / (y_0 * mu_e)
    RHO = C1 * mu_e * ((y_0 * Phi_discrete) ** 2 - 1) ** (1.5)

    print(R.shape, RHO.shape)

    return R / unit.to("metre"), RHO / unitsystem.density(unit)


def get_fermi_pressure_sound(rho, fermi_params, unit):
    """
    get_fermi_pressure_sound
    Returns a function that take rho in system unit and return (P, cs) in system units

    :param eos: eos dict
    :param unit: Shamrock unit
    """
    mu_e = fermi_params["mu_e"]

    length = unit.to("metre")
    density = unitsystem.density(unit)
    velocity = unitsystem.speed(unit)
    pressure = unitsystem.pressure(unit)
    return [
        su.P_fermi(rho * density, mu_e) / pressure,
        su.cs_fermi(rho * density, mu_e) / velocity,
    ]

    # return P_fermi, cs_fermi
    # elif eos["name"] == "polytropic":
    #     K = eos["values"]["K"]
    #     gamma = 1 + 1 / eos["values"]["n"]
    #     return lambda rho: [
    #         P_polytropic(rho * density, K, gamma) / pressure,
    #         cs_polytropic(rho * density, K, gamma) / velocity,
    #     ]


if __name__ == "__main__":
    import unitsystem

    codeearth = unitsystem.Unitsystem()
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
        [["m", "profile"], ["r", "profile"], ["u_c", "u_c"]], figsize=(12, 8)
    )
    figmass.suptitle(material)
    figmass.subplots_adjust(hspace=0.5)
    rhocenter_array = np.linspace(1.1 * rho0, 7 * rho0)
    mtot_array = []
    rmax_array = []
    for i, rhocenter in enumerate(rhocenter_array):
        tabx, tabrho = solve_hydrostatic_tillotson(kwargs_tillotson, rhocenter, T=0, unit=siunit)
        mtot = su.integrate_target([tabx, tabrho])
        rmax = np.max(tabx[tabrho > 10])
        mtot_array.append(mtot)
        rmax_array.append(rmax)

        if i % 3 == 0:
            line = axmass["profile"].plot(tabx, tabrho, label=f"{rhocenter/rho0:.1f}")[0]
            axmass["profile"].axvline(x=rmax, ls="--", alpha=0.4, color=line.get_color())

    axmass["profile"].set_ylabel(r"$\rho$")
    axmass["profile"].set_title(r"Density profile for different $\rho_{\rm center}/\rho_0$")
    axmass["m"].plot(rhocenter_array / rho0, mtot_array)
    axmass["m"].axhline(y=siunit.Mearth, color="blue", ls="--", alpha=0.5, label="Earth")
    axmass["m"].axhline(y=siunit.Mmoon, color="black", ls="--", alpha=0.5, label="Moon")
    axmass["m"].set_ylabel(r"$M_{\rm tot}$")
    axmass["m"].set_title(r"Total mass evolution with $\rho_{\rm center}/\rho_0$")

    axmass["r"].plot(rhocenter_array / rho0, rmax_array)
    axmass["r"].axhline(y=siunit.Rearth, color="blue", ls="--", alpha=0.5, label="Earth")
    axmass["r"].axhline(y=siunit.Rmoon, color="black", ls="--", alpha=0.5, label="Moon")
    axmass["r"].set_ylabel(r"$R_{\rm max}$")
    axmass["r"].set_title(r"Planet radius evolution with $\rho/\rho_0$ (RK45)")

    # u_c_func = solve_energy_cold(kwargs_tillotson)
    # rho_array = np.linspace(rho0, 6 * rho0, 100)
    rho_array, u_array = solve_energy_cold(kwargs_tillotson, unit=siunit)
    axmass["u_c"].plot(rho_array, u_array)
    axmass["u_c"].set_ylabel(r"$u_c$")
    axmass["u_c"].set_title(r"Cold energy curve with $\rho$")

    a, b, c = np.polyfit(rho_array, u_array, deg=2)
    axmass["u_c"].plot(
        rho_array,
        a * rho_array**2 + b * rho_array + c,
        color="black",
        ls="--",
        label=f"a{a:.4e} b{b:.4e} c{c:.4e}",
    )

    axmass["u_c"].axvline(x=rho0, color="black", ls=":", alpha=0.2, label=r"$\rho_0$")

    for label, ax in axmass.items():

        if label == "profile":
            ax.set_xlabel(r"$r$")
        elif label == "u_c":
            ax.set_xlabel(r"$\rho$")
        else:
            ax.set_xlabel(r"$\rho_{\rm center}/\rho_0$")

        ax.legend()

    plt.show()

# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./hydrostatic.py
