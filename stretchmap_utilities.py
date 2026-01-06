import numpy as np
import hydrostatic as hy


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
    """
    target[0] : r
    target[1] : rho
    """
    r = np.asarray(target[0])
    rho = np.asarray(target[1])

    # guard : formes attendues
    if r.ndim != 1 or rho.ndim != 1 or r.shape[0] != rho.shape[0]:
        raise ValueError("target must contains arrays of same length")

    integrand = rho * dS(r, dim)
    return np.trapezoid(integrand, r)


# Constantes fondamentales
# h = 6.62607015e-34  # Constante de Planck (J s)
# c = 2.99792458e8  # Vitesse de la lumière (m/s)
# m_e = 9.1093837015e-31  # Masse de l'électron (kg)
# m_p = 1.67262192369e-27  # Masse du proton (kg)
# Poids moléculaire moyen par électron (μ=2.0 pour les naines blanches typiques composées de C/O)
# mu = 2.0


# coeff_tpf = 2.04399770085e11
# coeff_tpf = 2.043997700850399e-3
# ALPHA = np.pow(3 / (8 * np.pi), 1.0 / 3) * coeff_tpf
# # coeff_p = 5.73180502079e-9
# coeff_p = 5731.80502078931e18
# BETA = coeff_p * np.pi / 3
ALPHA = 0.10064082802851738e-2  # /mu_e^{1/3}
BETA = 6002.332181706928e18


def tilde_pf(ALPHA, rho, mu_e):
    return ALPHA * rho ** (1.0 / 3.0) / mu_e ** (1.0 / 3)


def P_fermi(rho, mu_e):
    """
    Pressure in SI, with SI input

    :param rho: density in SI
    """
    tpf = tilde_pf(ALPHA, rho, mu_e)
    tpf2 = tpf * tpf
    P = BETA * (tpf * np.sqrt(tpf2 + 1) * (2 * tpf2 - 3) + 3 * np.asinh(tpf))
    return P


def cs_fermi(rho, mu_e):
    """
    Soundspeed in SI, with SI input

    :param rho: density in SI
    """
    tpf = tilde_pf(ALPHA, rho, mu_e)
    cs2 = (
        8
        * ALPHA
        * BETA
        / (3 * mu_e ** (1.0 / 3.0))
        * tpf**4
        / (np.sqrt(1 + tpf**2) * rho ** (2.0 / 3.0))
    )
    return np.sqrt(cs2)


def P_polytropic(rho, K, gamma):
    return K * rho**gamma


def cs_polytropic(rho, K, gamma):
    cs2 = gamma * P_polytropic(rho, K, gamma) / rho
    return np.sqrt(cs2)


if __name__ == "__main__":
    Msol = 1.989e30
    Rsol = 6.957e8
    Tsol = np.sqrt(Rsol**3 / (Msol * 6.67e-11))
    density = Msol / Rsol**3
    pressure = Msol / Rsol / Tsol**2
    velocity = Rsol / Tsol

    def stretch_fermi():
        # ###########################################
        y0 = 5
        mu_e = 2
        # ###########################################

        import matplotlib.pyplot as plt

        R, RHO = hy.solve_Chandrasekhar(y0, mu_e)  # SI

        pfermi_rho = P_fermi(RHO[:-1], mu_e) / pressure
        cs_rho = cs_fermi(RHO[:-1], mu_e) / velocity
        R = R[:-1] / Rsol
        RHO = RHO[:-1] / density

        fig, axs = plt.subplots(3)
        axs[0].plot(R, RHO)
        axs[1].plot(R, pfermi_rho)
        axs[2].plot(R, cs_rho)

        axs[0].set_title("Density")
        axs[1].set_title("Pressure")
        axs[2].set_title("Soundspeed")
        axs[0].set_ylabel(r"$\rho$")
        axs[1].set_ylabel(r"$P$")
        axs[2].set_ylabel(r"$c_s$")

        for ax in axs:
            ax.set_xlabel(r"$r/R_{\odot}$")
        fig.subplots_adjust(hspace=0.6, left=0.3, right=0.7)
        plt.show()

        import pandas as pd

        df = pd.DataFrame({"r": R, "rho_target": RHO, "P_target": pfermi_rho, "cs_target": cs_rho})
        df.to_csv(f"rhotarget_y0-{y0}_mue-{mu_e}.csv", index=False)

        print(Msol, Rsol, Tsol)
        print(pressure, density, velocity)
        c = 2.99792458
        h = 6.62607015
        me = 9.1093837
        mp = 1.67262192
        G = 6.67430

        c_exp = 8
        h_exp = -34
        me_exp = -31
        mp_exp = -27
        G_exp = -11

        ALPHA = (3 / (8 * np.pi)) ** (1.0 / 3) * h / (mp ** (1.0 / 3.0) * me * c)
        _3ALPHA_exp = 3 * h_exp - mp_exp - 3 * me_exp - 3 * c_exp

        print(f"ALPHA = {ALPHA} *e{_3ALPHA_exp}/3 ")

        BETA = (np.pi / 3) * me**4 * c**5 / h**3
        BETA_exp = me_exp * 4 + c_exp * 5 - h_exp * 3

        print(f"BETA = {BETA} *e{BETA_exp} ")

        C1 = (me * c / h) ** 3 * mp * (8 * np.pi / 3)  # a mutiplier par mu_e
        C1_exp = (me_exp + c_exp - h_exp) * 3 + mp_exp

        print(f"C1 = {C1} *e{C1_exp} ")

        RA = (1 / C1) * np.sqrt(2 * BETA / (np.pi * G))
        _2RA_exp = -2 * C1_exp + (BETA_exp - G_exp)
        print(f"RA = {RA} *e{_2RA_exp}/2 ")  # a diviser par (y_0*mu_e)
