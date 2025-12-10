import numpy as np
import stretchmap_utilities as su
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


if __name__ == "__main__":
    plt.close("all")

    ###########################################
    y0_value = 1.5  # Exemple : paramètre de relativité
    mu_e = 2
    ###########################################

    R, RHO = su.solve_Chandrasekhar(y0_value)  # SI

    pfermi_rho = su.P_fermi(RHO, mu_e) / su.pressure
    cs_rho = su.cs_fermi(RHO[:-1], mu_e) / su.velocity
    R /= su.Rsol
    fig, axs = plt.subplots(3)

    titles = ["Density", "Pressure", "Soundspeed"]
    labelsx = [r"$r/R_\odot$", r"$r/R_\odot$", r"$r/R_\odot$"]
    labelsy = [r"$\rho$", r"$P$", r"$c_s$"]
    colors = ["blue", "green", "magenta"]
    X = [R, R, R[:-1]]
    Y = [RHO, pfermi_rho, cs_rho]
    for i, ax in enumerate(axs):
        ax.plot(X[i], Y[i], label=f"$y_0={y0_value}$", color=colors[i])
        ax.set_xlabel(labelsx[i])
        ax.set_ylabel(labelsy[i])
        ax.set_title(titles[i])

    figeos, axeos = plt.subplots()
    rho = np.linspace(0.01, 1, 100)
    axeos.plot(rho, su.P_fermi(rho, mu_e), color="green", label="pressure")
    axeos.plot(rho, su.cs_fermi(rho, mu_e), color="magenta", label="soundspeed")
    axeos.set_xlabel(r"$\rho$")
    # axeos.set_ylabel("rho")

    axes = [*axs, axeos]
    for ax in axes:
        ax.legend()

    # for fg in [fig, figeos]:
    #     fg.tight_layout(pad=0.4, w_pad=1, h_pad=1.0)
    fig.subplots_adjust(left=0.3, right=1 - 0.3, hspace=0.5)
    fig.suptitle(f"$y_0={y0_value}, \\mu_e={mu_e} $")

    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

# sol.t_events[0] donne l'emplacement du rayon de surface eta_s
# sol.y[0] est la solution pour Phi, sol.y[1] est la solution pour Phi'
