import numpy as np
import stretchmap_utilities as su
import matplotlib.pyplot as plt

# # Constantes fondamentales
# h = 6.62607015e-34  # Constante de Planck (J s)
# c = 2.99792458e8  # Vitesse de la lumière (m/s)
# m_e = 9.1093837015e-31  # Masse de l'électron (kg)
# m_p = 1.67262192369e-27  # Masse du proton (kg)
# # Poids moléculaire moyen par électron (μ=2.0 pour les naines blanches typiques composées de C/O)
# mu = 2.0
# G = 6.67e-11

# Msol = 1.989e30
# Rsol = 6.957e8


if __name__ == "__main__":
    plt.close("all")

    ###########################################
    y0_value = 1.5  # Exemple : paramètre de relativité
    mu_e = 2
    ###########################################
    y0s = np.linspace(1.2, 14, 20)
    mtots = []
    rmaxs = []
    for y0_value in y0s:

        R, RHO = su.solve_Chandrasekhar(y0_value, mu_e)  # SI

        pfermi_rho = su.P_fermi(RHO, mu_e) / su.pressure
        cs_rho = su.cs_fermi(RHO[:-1], mu_e) / su.velocity
        R = R[:-1] / su.Rsol
        RHO = RHO[:-1] / su.density

        mtot = su.integrate_target([R, RHO])
        print(f"y0 {y0_value} mtot {mtot:.1e}")
        print(RHO)
        mtots.append(mtot)
        rmaxs.append(np.max(R))

    figcheck, axcheck = plt.subplots(2)
    axcheck[0].plot(y0s, mtots, marker="o")
    axcheck[0].set_xlabel(r"$y_0$")
    axcheck[0].set_ylabel(r"$M/M_{\odot}$")
    axcheck[1].plot(y0s, np.array(rmaxs) * su.Rsol, marker="o")
    axcheck[1].set_xlabel(r"$y_0$")
    axcheck[1].set_ylabel(r"$R$")
    plt.show()
    exit()

    fig, axs = plt.subplots(3)
    titles = ["Density", "Pressure", "Soundspeed"]
    labelsx = [r"$r/R_\odot$", r"$r/R_\odot$", r"$r/R_\odot$"]
    labelsy = [r"$\rho$", r"$P$", r"$c_s$"]
    colors = ["blue", "green", "magenta"]
    X = [R, R, R[:-1]]
    Y = [RHO / su.density, pfermi_rho, cs_rho]
    for i, ax in enumerate(axs):
        ax.plot(X[i], Y[i], label=f"$y_0={y0_value}$", color=colors[i])
        ax.set_xlabel(labelsx[i])
        ax.set_ylabel(labelsy[i])
        ax.set_title(titles[i])
        ax.ticklabel_format(scilimits=(-1, 1))

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
