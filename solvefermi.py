import numpy as np
import stretchmap_utilities
import matplotlib.pyplot as plt

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


def P_fermi(rho):
    pf = rho ** (1.0 / 3.0)
    pf2 = pf * pf
    return pf * (np.sqrt(pf2) + 1) * (2 * pf2 - 3) + 3 * np.asinh(pf)


def cs_fermi(rho):
    pf = rho ** (1.0 / 3.0)
    return pf**4 / (np.sqrt(1 + pf**2) * rho ** (2.0 / 3.0))


if __name__ == "__main__":

    ###########################################
    y0_value = 1.5  # Exemple : paramètre de relativité
    ###########################################

    R, RHO = stretchmap_utilities.solve_Chandrasekhar(y0_value)

    pfermi_rho = P_fermi(RHO)
    cs_rho = cs_fermi(RHO[:-1])
    print(cs_rho)
    fig, ax = plt.subplots()
    ax.plot(R, RHO, label="density")
    ax.plot(R, pfermi_rho * np.max(RHO) / np.max(pfermi_rho), label="pressure")
    ax.plot(R[:-1], cs_rho * np.max(RHO) / np.max(cs_rho), label="soundspeed")
    ax.set_xlabel("R")

    figeos, axeos = plt.subplots()
    rho = np.linspace(0.01, 1, 100)
    axeos.plot(rho, P_fermi(rho), label="pressure")
    axeos.plot(rho, cs_fermi(rho), label="soundspeed")
    axeos.set_xlabel("rho")
    # axeos.set_ylabel("rho")

    axes = [ax, axeos]
    for ax in axes:
        ax.legend()
    plt.show()

# sol.t_events[0] donne l'emplacement du rayon de surface eta_s
# sol.y[0] est la solution pour Phi, sol.y[1] est la solution pour Phi'
