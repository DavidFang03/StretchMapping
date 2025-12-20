import matplotlib.pyplot as plt
import numpy as np
import hydrostatic as hy


def phase_diagram(kwargs):
    # --- Paramètres de configuration ---
    rho0 = kwargs.get("rho0", 2700)
    u_iv = kwargs.get("u_iv", 5.0e6)
    u_cv = kwargs.get("u_cv", 15.0e6)

    # Limites du graphique pour la visualisation
    max_rho = 4 * rho0 / 3
    max_u = 4 * u_cv / 3

    # Création de la figure
    fig, ax = plt.subplots(figsize=(14, 9))  # Format large pour placer les équations

    # --- Définition des zones (remplissage couleur) ---

    # 1. Région Compressée (rho >= rho0)
    ax.fill_between([rho0, max_rho], 0, max_u, color="#e0e0e0", alpha=0.6)

    # 2. Région Froide / Expansion (rho < rho0, u < u_iv)
    ax.fill_between([0, rho0], 0, u_iv, color="#cce5ff", alpha=0.6)

    # 2. Région TRES Froide / Expansion (rho < rhoIV, u < u_iv)
    # ax.fill_between([0, rhoiv], 0, u_iv, color="#234466", alpha=0.6)

    # 3. Région Chaude / Vapeur (rho < rho0, u > u_cv)
    ax.fill_between([0, rho0], u_cv, max_u, color="#ffcccc", alpha=0.6)

    # 4. Région de Transition (rho < rho0, u_iv < u < u_cv)
    ax.fill_between([0, rho0], u_iv, u_cv, color="#e6ccff", alpha=0.6)

    # --- Lignes de séparation ---
    # ax.axvline(x=rhoiv, color="black", linestyle="-", linewidth=2)
    ax.axvline(x=rho0, color="black", linestyle="-", linewidth=2)
    ax.axhline(y=u_iv, xmax=rho0 / max_rho, color="blue", linestyle="--", linewidth=1.5)
    ax.axhline(y=u_cv, xmax=rho0 / max_rho, color="red", linestyle="--", linewidth=1.5)

    # --- Ajout des Équations (LaTeX) ---

    # Style de base pour les boîtes de texte
    props = dict(
        boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray", pad=1
    )

    # EQ 1: Compressé
    eq_1 = (
        r"$\bf{1.\ COMPRESSION}$"
        + "\n"
        + r"($\rho \geq \rho_0$)"
        + "\n\n"
        + r"$P = \left[ a + \frac{b}{1 + \frac{u}{E_0 \eta^2}} \right] \rho u + A \chi + B \chi^2$"
        + "\n with\n"
        + r"$\eta = \rho/\rho_0$"
        + "\n"
        + r"$\chi = \eta - 1$"
    )
    ax.text(
        (rho0 + max_rho) / 2,
        max_u / 2,
        eq_1,
        ha="center",
        va="center",
        fontsize=11,
        bbox=props,
    )

    # EQ 2: Froid / Détendu
    # eq_2a = (
    #     r"$\bf{2.\ LOW \ ENERGY\ EXPANSION}$"
    #     + "\n"
    #     + r"($\rho < \rho_0,\ u < u_{iv}$)"
    #     + "\n\n"
    #     + r"$P = \left[ a + \frac{b}{1 + \frac{u}{E_0 \eta^2}} \right] \rho u + A \chi$"
    # )
    # ax.text(rhoiv / 2, u_iv / 2, eq_2a, ha="center", va="center", fontsize=10, bbox=props)

    # EQ 2: Froid / Détendu
    eq_2 = (
        r"$\bf{2.\ COLD\ EXPANSION}$"
        + "\n"
        + r"($\rho < \rho_0,\ u < u_{iv}$)"
        + "\n\n"
        + r"$P = \left[ a + \frac{b}{1 + \frac{u}{E_0 \eta^2}} \right] \rho u + A \chi + B\chi^2$"
    )
    ax.text(
        (rho0) / 2,
        u_iv / 2,
        eq_2,
        ha="center",
        va="center",
        fontsize=10,
        bbox=props,
    )

    # EQ 3: Chaud / Vapeur
    # Note: Formule simplifiée visuellement pour tenir dans la boite
    eq_3 = (
        r"$\bf{3.\ HOT\ EXPANSION}$"
        + "\n"
        + r"($\rho < \rho_0,\ u > u_{cv}$)"
        + "\n\n"
        + r"$P = a\rho u + \left[ \frac{b\rho u}{1 + \frac{u}{E_0 \eta^2}} + A\chi e^{-\beta X} \right] e^{-\alpha X^2}$"
        + "\n\n"
        + r"with $X = \left( \frac{\rho_0}{\rho} - 1 \right)$"
    )
    ax.text(
        rho0 / 2,
        (u_cv + max_u) / 2,
        eq_3,
        ha="center",
        va="center",
        fontsize=10,
        bbox=props,
    )

    # EQ 4: Transition
    eq_4 = (
        r"$\bf{4.\ MIXED \ PHASE \ STATE}$"
        + "\n"
        + r"($u_{iv} < u < u_{cv}$)"
        + "\n\n"
        + r"$P = \frac{(u - u_{iv}) P_{hot} + (u_{cv} - u) P_{cold}}{u_{cv} - u_{iv}}$"
    )
    ax.text(
        rho0 / 2,
        (u_iv + u_cv) / 2,
        eq_4,
        ha="center",
        va="center",
        fontsize=10,
        bbox=props,
    )

    # --- Mise en forme des axes ---
    ax.set_xlim(0, max_rho)
    ax.set_ylim(0, max_u)
    ax.set_xlabel(r"Density $\rho$ (kg/m$^3$)", fontsize=14)
    ax.set_ylabel(r"Internal energy $u$ (J/kg)", fontsize=14)
    ax.set_title(r"Phase Diagram - Tillotson EoS in Shamrock", fontsize=16, pad=20)
    fig.suptitle("David FANG")

    # Annotations sur les axes
    ax.text(rho0, -max_u * 0.03, r"$\rho_0$", ha="center", fontsize=12, color="black")
    ax.text(-max_rho * 0.03, u_iv, r"$u_{iv}$", va="center", fontsize=12, color="blue")
    ax.text(-max_rho * 0.03, u_cv, r"$u_{cv}$", va="center", fontsize=12, color="red")

    ax.set_xticks([0])
    ax.set_yticks([0])

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
    kwargs_tillotson = {
        "rho0": 7.8e3,  # kg/m^3
        "E0": 0.095e8,  # J/kg (Spécifique energy of sublimation approx)
        "a": 0.5,
        "b": 1.5,
        "A": 1.279e11,  # Pa (Bulk modulus A)
        "B": 1.05e11,  # Pa (Non-linear modulus B)
        "alpha": 5.0,
        "beta": 5.0,
        "u_iv": 0.024e8,
        "u_cv": 0.0867e8,
        "u_int": 1e5,  # Energie interne initiale (J/kg) - "Froid" (Capa thermique ~ 4e2 -> C\deltaT ~1e5 < u_iv)
        "rho_center": 8000.0,  # On force une densité centrale > rho0 pour voir le profil
    }

    fig = phase_diagram(kwargs_tillotson)
    fig_Prho = P_of_rho(kwargs_tillotson)
    plt.show()
