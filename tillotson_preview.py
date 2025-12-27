import matplotlib.pyplot as plt
import numpy as np
import hydrostatic as hy

fs = 24
fs_box = 24
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica", "font.size": fs})


def phase_diagram():
    # --- Paramètres de configuration ---
    rho0 = 2
    u_iv = 1
    u_cv = 2

    # Limites du graphique pour la visualisation
    max_rho = 3.5
    max_u = 3

    # Création de la figure
    fig, ax = plt.subplots(figsize=(14, 10))  # Format large pour placer les équations

    # --- Définition des zones (remplissage couleur) ---

    # 1. Région Compressée (rho >= rho0)
    ax.fill_between([rho0, max_rho], 0, max_u, color="#cce5ff", alpha=0.6)
    # ax.fill_between([rho0, max_rho], 0, max_u, color="#e0e0e0", alpha=0.6)

    # 2. Région Froide / Expansion (rho < rho0, u < u_iv)
    ax.fill_between([0, rho0], 0, u_iv, color="#cce5ff", alpha=0.6)

    # 2. Région TRES Froide / Expansion (rho < rhoIV, u < u_iv)
    # ax.fill_between([0, rhoiv], 0, u_iv, color="#234466", alpha=0.6)

    # 3. Région Chaude / Vapeur (rho < rho0, u > u_cv)
    ax.fill_between([0, rho0], u_cv, max_u, color="#ffcccc", alpha=0.6)

    ax.fill_between([0, rho0], u_iv, u_cv, color="black", alpha=0.6)

    # 4. Région de Transition (rho < rho0, u_iv < u < u_cv)
    u_space = np.linspace(u_iv, u_cv, 20)
    for i in range(len(u_space) - 1):
        ax.fill_between(
            [0, rho0],
            u_space[i],
            u_space[i + 1],
            color="#cce5ff",
            alpha=1 - (i / len(u_space)),
        )
        ax.fill_between(
            [0, rho0],
            u_space[i],
            u_space[i + 1],
            color="#ffcccc",
            alpha=i / len(u_space),
        )

    # --- Lignes de séparation ---
    # ax.axvline(x=rhoiv, color="black", linestyle="-", linewidth=2)
    ax.axvline(x=rho0, ymax=u_iv / max_u, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(x=rho0, ymin=u_iv / max_u, color="black", linestyle="-", linewidth=3)
    ax.axhline(y=u_iv, xmax=rho0 / max_rho, color="blue", linestyle="--", linewidth=1.5)
    ax.axhline(y=u_cv, xmax=rho0 / max_rho, color="red", linestyle="--", linewidth=1.5)

    # --- Ajout des Équations (LaTeX) ---

    # Style de base pour les boîtes de texte
    props = dict(
        boxstyle="round", facecolor="white", alpha=0.0, edgecolor="gray", pad=1
    )

    # EQ 1: Compressé
    eq_1 = (
        r"$\bf{\ COMPRESSION}$"
        # + "\n"
        # + r"($\rho \geq \rho_0$)"
        + "\n"
        + r"$P^{(\rm cold)} = \left[ a + \frac{b}{1 + \frac{u}{E_0 \eta^2}} \right] \rho u + A \chi + B \chi^2$"
    )
    ax.text(
        (rho0 + max_rho) / 2,
        max_u / 2,
        eq_1,
        ha="center",
        va="center",
        fontsize=fs_box,
        bbox=props,
    )

    # EQ 2: Froid / Détendu
    eq_2 = (
        r"$\bf{\ COLD\ EXPANSION}$"
        # + "\n"
        # + r"($\rho < \rho_0,\ u < u_{iv}$)"
        + "\n"
        + r"$P^{(\rm cold)} = \left[ a + \frac{b}{1 + \frac{u}{E_0 \eta^2}} \right] \rho u + A \chi + B\chi^2$"
    )
    ax.text(
        (rho0) / 2,
        u_iv / 2,
        eq_2,
        ha="center",
        va="center",
        fontsize=fs_box,
        bbox=props,
    )

    # EQ 3: Chaud / Vapeur
    # Note: Formule simplifiée visuellement pour tenir dans la boite
    eq_3 = (
        r"$\bf{\ HOT\ EXPANSION}$"
        # + "\n"
        # + r"($\rho < \rho_0,\ u > u_{cv}$)"
        + "\n"
        + r"$P^{(\rm hot)} = a\rho u + \left[ \frac{b\rho u}{1 + \frac{u}{E_0 \eta^2}} + A\chi e^{-\beta X} \right] e^{-\alpha X^2}$"
    )
    ax.text(
        rho0 / 2,
        (u_cv + max_u) / 2,
        eq_3,
        ha="center",
        va="center",
        fontsize=fs_box,
        bbox=props,
    )

    # EQ 4: Transition
    eq_4 = (
        r"$\bf{\ MIXED \ PHASE \ STATE}$"
        # + "\n"
        # + r"($u_{iv} < u < u_{cv}$)"
        + "\n\n"
        + r"$P = \frac{(u - u_{iv}) P^{(\rm hot)} + (u_{cv} - u) P^{(\rm cold)}}{u_{cv} - u_{iv}}$"
    )
    ax.text(
        rho0 / 2,
        (u_iv + u_cv) / 2,
        eq_4,
        ha="center",
        va="center",
        fontsize=fs_box,
        bbox=props,
    )

    ax.text(
        0.88 * max_rho,
        0.03 * max_rho,
        r"$X = \left( \frac{\rho_0}{\rho} - 1 \right)$"
        + "\n"
        + r"$\eta = \rho/\rho_0$"
        + "\n"
        + r"$\chi = \eta - 1$",
        fontsize=16,
        alpha=0.6,
    )

    # --- Mise en forme des axes ---
    ax.set_xlim(0, max_rho)
    ax.set_ylim(0, max_u)
    ax.set_xlabel(r"Density $\rho$", fontsize=fs)
    ax.set_ylabel(r"Internal energy $u$", fontsize=fs)
    # ax.tick_params(axis="y", which="major", pad=30)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 40
    ax.set_title(r"Phase Diagram - Tillotson EoS in Shamrock", fontsize=16, pad=20)
    fig.suptitle("David FANG")

    # Annotations sur les axes
    ax.text(rho0, -max_u * 0.03, r"$\rho_0$", ha="center", fontsize=fs, color="black")
    ax.text(-max_rho * 0.04, u_iv, r"$u_{iv}$", va="center", fontsize=fs, color="blue")
    ax.text(-max_rho * 0.04, u_cv, r"$u_{cv}$", va="center", fontsize=fs, color="red")
    # ax.text(0, 0, r"$0$", va="center", fontsize=fs, color="black")

    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    return fig


def P_of_rho(kwargs):
    rho0 = kwargs.get("rho0", 7.8e3)
    u_int = kwargs.get("u_int", 0.5 * 5.0e6)
    RHO = np.linspace(0, 2 * rho0, 200)[1:]
    P, CS = hy.get_tillotson_pressure_sound(RHO, u_int, kwargs)
    fig, axs = plt.subplots(2)
    axs[0].plot(RHO, P)
    axs[0].set_ylabel(r"$P$")
    axs[1].plot(RHO, CS)
    axs[1].set_ylabel(r"$c_s$")

    for ax in axs:
        ax.axvline(x=rho0, color="black", ls="--", lw="1", alpha=0.4)
        ax.axhline(y=0, color="black", ls="--", lw="1", alpha=0.4)
        ax.set_xlabel(r"$\rho$")
    return fig


if __name__ == "__main__":

    fig = phase_diagram()
    fig.savefig("tillotson_preview.png")
    # from hydrostatic import Tillotson_parameters_Granite
    # fig_Prho = P_of_rho(Tillotson_parameters_Granite)
    # plt.show()
