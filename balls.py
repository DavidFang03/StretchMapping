import numpy as np
import sham_utilities as shu
import stretchmap_utilities as stu
import hydrostatic as hy
import unitsystem
import matplotlib.pyplot as plt

import shamrock

shamrock.enable_experimental_features()

style_target = "dashed"
colors = {
    "density": "blue",
    "energy": "orange",
    "pressure": "green",
    "soundspeed": "magenta",
}
ids = {
    "density": "rho",
    "energy": "uint",
    "pressure": "pressure",
    "soundspeed": "soundspeed",
}

Tillotson_parameters_Fe = {
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
}


class EoS:
    """
    For Tillotson for example:
        | id="Tillotson"
        | params=...Tillotson parameters...
        | details=...Stretchmapping parameters (rhocenter here))
    details must contain
        | "material" : Fe e.g
    """

    def __init__(self, id, unit, details):
        self.id = id
        self.unit = unit
        self.details = details
        if id == "tillotson":
            self.setup_Tillotson_EoS()
        else:
            raise NotImplementedError()

    def setup_Tillotson_EoS(self):
        self.material = self.details.get("material", "Fe")

        match self.material:
            case "Fe":
                params = Tillotson_parameters_Fe
        self.params = hy.adimension(params, self.unit)

    def solve_hydrostatic(self, values):
        """
        must return tabx, tabrho
        """
        if self.id == "tillotson":
            rho_center = values["rho_center"]
            u_int = values["u_int"]
            tabx, tabrho = hy.solve_hydrostatic_tillotson(
                self.params, rho_center, u_int, self.unit
            )
        return tabx, tabrho

    def get_p_and_cs(self, rho, u_int):
        if self.id == "tillotson":
            return hy.get_tillotson_pressure_sound(rho, u_int, self.params)

    def tojson(self):
        def format_dict(dict):
            formated_dict = {}
            for key, value in dict.items():
                if isinstance(value, float):
                    formated_dict[key] = f"{value:.1e}"
                else:
                    formated_dict[key] = value

        eos_json = {
            "id": self.id,
            "params": format_dict(self.params),
            "details": format_dict(self.details),
        }
        if self.id == "tillotson":
            eos_json["material"] = self.material


class Tillotson_Ball:
    def __init__(self, center, v_xyz, N_target, rho_center, u_int, eos):
        """
        Can be created before setup

        :param center:
        :param v_xyz:
        :param N_target:
        :param eos: EoS instance
        :param rho_center:  in SI
        :param u_int:       in SI
        :param unit: Unit instance
        """
        self.center = center
        self.v_xyz = v_xyz
        self.N_target = int(N_target)

        rho_center = rho_center / unitsystem.density(eos.unit)
        u_int = u_int / unitsystem.energy(eos.unit)

        self.data = {
            "r": None,
            "density": None,
            "energy": None,
            "pressure": None,
            "soundspeed": None,
        }
        (
            self.data["r"],
            self.data["density"],
        ) = eos.solve_hydrostatic({"rho_center": rho_center, "u_int": u_int})
        self.mtot_target = stu.integrate_target([self.data["r"], self.data["density"]])

        xmax = float(np.max(self.data["r"]))
        part_vol = ((2 * xmax) ** 3) / N_target
        HCP_PACKING_DENSITY = 0.74
        part_vol_lattice = HCP_PACKING_DENSITY * part_vol
        self.dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)

        self.xmax = xmax
        # self.bmin = self.center - np.array([xmax, xmax, xmax])
        self.bmin = (center[0] - xmax, center[1] - xmax, center[2] - xmax)
        self.bmax = (center[0] + xmax, center[1] + xmax, center[2] + xmax)
        # self.bmax = self.center + np.array([-xmax, -xmax, -xmax])
        self.pmass = self.mtot_target / (N_target / 2)

        self.data["energy"] = [u_int for _ in self.data["r"]]
        self.data["pressure"], self.data["soundspeed"] = eos.get_p_and_cs(
            self.data["density"], self.data["energy"]
        )

    def show_profile(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.data["r"], self.data["density"], marker="x")
        ax.set_xlabel("$r$")
        ax.set_ylabel("$\\rho$")
        plt.show()


class Setup:
    def __init__(self, SG, eos, balls, nb_dumps, tf, overwrite):
        self.SG = SG
        self.eos = eos
        self.balls = balls
        self.unit = eos.unit
        self.ready_toloop = False
        self.fig = None
        self.nb_dumps = nb_dumps
        self.tf = tf

        self.init_model(overwrite=overwrite)

    def init_model(self, overwrite):
        """
        Roadmap:
            - initialize, ctx, model (kernel), cfg (solver config)
            - artificial viscosity
            - add balls (and compute eps_plummer)
            - self-gravity
            - EOS
            - simulation size
            - CFL
        This does NOT :
            - set particle mass
            - stretchmap
        """
        if not shamrock.sys.is_initialized():
            shamrock.change_loglevel(1)
            shamrock.sys.init("0:0")

        self.ctx = shamrock.Context()
        self.ctx.pdata_layout_new()
        self.model = shamrock.get_Model_SPH(
            context=self.ctx, vector_type="f64_3", sph_kernel="M4"
        )
        self.cfg = self.model.gen_default_config()

        # -------------------------------------------------------------------------------
        # artificial viscocity  ----------------------------------------------------------
        self.cfg.set_artif_viscosity_VaryingCD10(
            alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
        )
        # _______________________________________________________________________________

        # -------------------------------------------------------------------------------
        # Self-gravity ------------------------------------------------------------------
        if self.SG == "mm":
            self.cfg.set_self_gravity_mm(
                order=5, opening_angle=0.5, reduction_level=3
            )  #! self-gravity
        elif self.SG == "fmm":
            self.cfg.set_self_gravity_fmm(
                order=5, opening_angle=0.5, reduction_level=3
            )  #! self-gravity
        else:
            raise NotImplementedError()
        self.eps_plummer = self.get_eps_plummer()
        self.cfg.set_softening_plummer(epsilon=self.eps_plummer)

        # // if self.eos.id == "fermi":
        # // self.cfg.set_eos_fermi(**self.eos.params)
        # // elif self.eos.id == "polytropic":
        # // self.cfg.set_eos_polytropic(**self.params)
        if self.eos.id == "tillotson":
            self.cfg.set_eos_tillotson(**self.eos.params)
        else:
            raise NotImplementedError()
        # _______________________________________________________________________________

        # -------------------------------------------------------------------------------
        # units  ------------------------------------------------------------------------
        self.cfg.set_units(self.unit)
        # _______________________________________________________________________________

        self.model.set_solver_config(self.cfg)
        self.model.init_scheduler(int(1e7), 1)

        # -------------------------------------------------------------------------------
        # resize simulation box ---------------------------------------------------------
        sbmin = (-1, -1, -1)
        sbmax = (1, 1, 1)
        self.model.resize_simulation_box(sbmin, sbmax)
        # _______________________________________________________________________________

        self.add_balls()

        # -------------------------------------------------------------------------------
        # CFL ---------------------------------------------------------------------------
        C_cour = 0.1
        C_force = 0.1
        self.model.set_cfl_cour(C_cour)
        self.model.set_cfl_force(C_force)
        # _______________________________________________________________________________

        self.dump_prefix = self.gen_dump_prefix()
        self.folder_path = self.handle_dump(overwrite=overwrite)

    def get_eps_plummer(self):
        eps_plummer = np.inf
        for ball in self.balls:
            hfact = self.model.get_hfact()  # ou la valeur voulue
            mpart_target = ball.mtot_target / (ball.N_target / 2)  # masse par particule
            hmin = hfact * (mpart_target / np.max(ball.data["density"])) ** (1.0 / 3.0)
            if eps_plummer > hmin:
                eps_plummer = hmin  # h min à peu près
        return eps_plummer

    def add_balls(self):
        """
        :param center:
        :param v_xyz:
        :param N_target:
        :param eos:
        :param rho_center:
        :param u_int:
        """
        for ball in self.balls:
            # TODO During impact, eps_plummer may have to be smaller ?

            setup = self.model.get_setup()
            hcp = setup.make_generator_lattice_hcp(ball.dr, ball.bmin, ball.bmax)

            stretched_hcp = setup.make_modifier_stretch_mapping(
                parent=hcp,
                system="spherical",
                axis="r",
                box_min=ball.bmin,
                box_max=ball.bmax,
                tabx=ball.data["r"],
                tabrho=ball.data["density"],
                mtot=ball.mtot_target,
            )

            setup.apply_setup(stretched_hcp)

    def gen_dump_prefix(self):
        balls_nb = len(self.balls)
        dump_prefix = f"1ball_" if balls_nb == 1 else f"{balls_nb}balls_"
        dump_prefix += f"{self.eos.id}_"
        dump_prefix += f"{int(self.balls[0].N_target/1000)}k_"
        dump_prefix += f"{self.SG}_"
        dump_prefix += "cd10_"
        return dump_prefix

    def handle_dump(self, overwrite):
        import os
        import glob

        try:
            os.mkdir("outputs")
        except OSError as error:
            print("ok outputs exist")

        last_eventual_folder_path = shu.get_last_folder(self.dump_prefix)
        if overwrite and os.path.isdir(last_eventual_folder_path):
            folder_path = shu.get_last_folder(self.dump_prefix)
            user_agree = input(
                f"Will remove the entire {folder_path} folder ({len(glob.glob(f"{folder_path}/*.sham"))} dumps) (y/n)"
            )
            if user_agree == "y":

                for f in glob.glob(f"{folder_path}/*"):
                    os.remove(f)
        else:
            folder_path = shu.get_new_folder(self.dump_prefix)
            try:
                os.mkdir(folder_path)
            except OSError as error:
                print(
                    f"{folder_path} Directory already exists, no need to mkdir a new one."
                )

        command = f"cp {os.path.abspath(__file__)} {folder_path}/{self.dump_prefix}.py"
        print("executing", command)
        os.system(command)
        print("Ok let's go with", folder_path)
        return folder_path

    def get_free_fall_time(self):
        """
        For 1 or 2 balls
        """
        match len(self.balls):
            case 0:
                raise Exception("Setup is empty")
            case 1:
                ball = self.balls[0]
                return np.sqrt(ball.xmax**3 / ball.mtot_target)
            case 2:  # TODO Check this
                distance = np.linalg.norm(self.balls[0].center - self.balls[1].center)
                maxmass = max(self.balls[0].mtot_target, self.balls[1].mtot_target)
                return np.sqrt(distance**3 / maxmass)
        return 1

    def dump(self, dump_path):  # TODO directly to .vtk ?
        self.model.dump(dump_path)
        print(f"Dumped {dump_path}")

    def firstplot(self):
        self.fig, axs = plt.subplots(2)
        self.axes = {
            "density": axs[0],
            "energy": axs[0].twinx(),
            "soundspeed": axs[1],
            "pressure": axs[1].twinx(),
        }
        self.graphs = {
            "density": None,
            "energy": None,
            "soundspeed": None,
            "pressure": None,
        }
        data = self.ctx.collect_data()
        data["r"] = np.linalg.norm(data["xyz"], axis=1)
        data["rho"] = (
            self.model.get_particle_mass()
            * (self.model.get_hfact() / data["hpart"]) ** 3
        )
        if len(data["r"]) >= 1e4:
            self.markevery = len(data["r"]) // 1e4
        else:
            self.markevery = 1

        for quantity in ["density", "energy", "soundspeed", "pressure"]:
            self.axes[quantity].plot(
                self.balls[0].data["r"],
                self.balls[0].data[quantity],
                label="target",
                ls=style_target,
                color=colors[quantity],
            )
            self.graphs[quantity] = self.axes[quantity].plot(
                data["r"],
                data[ids[quantity]],
                marker="o",
                markersize=3,
                ls="",
                label="data",
                markevery=self.markevery,
                color=colors[quantity],
            )[0]

        print("energy" in self.axes)
        self.axes["density"].set_ylabel("$\\rho$")
        self.axes["energy"].set_ylabel("$u$")
        self.axes["soundspeed"].set_ylabel("$c_s$")
        self.axes["pressure"].set_ylabel("$p$")
        for ax in self.axes.values():
            ax.legend()
            ax.set_xlabel("$r$")

    def update_plot(self):
        data = self.ctx.collect_data()
        data["r"] = np.linalg.norm(data["xyz"], axis=1)
        data["rho"] = (
            self.model.get_particle_mass()
            * (self.model.get_hfact() / data["hpart"]) ** 3
        )
        for quantity in ["density", "energy", "soundspeed", "pressure"]:
            self.graphs[quantity].set_ydata(data[ids[quantity]])

    def ready_set_go(self, rescale=False):
        """
        Roadmap:
            - First dump #//(and first plot?)
            - One timestep with dt=0
            - # ?Rescale stretchmapping? Idk how to do it if there are multiple balls.
        """
        self.write_json_params()
        newpath_withoutext = shu.gen_new_path_withoutext(self.dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        self.dump(dump_path)

        self.model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
        self.model.evolve_once_override_time(0.0, 0.0)
        self.model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))

        newpath_withoutext = shu.gen_new_path_withoutext(self.dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        img_path = f"{newpath_withoutext}.png"
        self.dump(dump_path)
        self.firstplot()
        self.fig.savefig(img_path)
        self.ready_toloop = True
        print("Dump success: ready to smash balls")

    def loop(self, t_stops):
        if not self.ready_toloop:
            raise Exception("Setup is not ready")
        for i, t in enumerate(t_stops):
            print(f"looping {i}, still {len(t_stops)-1} to go")
            self.model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
            self.model.evolve_until(t)
            self.model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))

            newpath_withoutext = shu.gen_new_path_withoutext(self.dump_prefix)
            dump_path = f"{newpath_withoutext}.sham"
            img_path = f"{newpath_withoutext}.png"
            self.dump(dump_path=dump_path)  # **BEFORE** plotting
            self.update_plot()
            self.fig.savefig(img_path)

    def write_json_params(self):
        import json

        jsonparams = {
            "nb_dumps": self.nb_dumps,
            "tf": f"{self.tf:.1e}",
            "pmass": [f"{ball.pmass:.1e}" for ball in self.balls],
            "mtot_target": [f"{ball.mtot_target:.1e}" for ball in self.balls],
            "dr": [f"{ball.dr:.1e}" for ball in self.balls],
            "xmax": [f"{ball.xmax:.1e}" for ball in self.balls],
            "eos": self.eos.tojson(),
            "target": "solve_ivp(RK45)",
            "SG": self.SG,
            "eps_plummer": f"{self.eps_plummer:.1e}",
        }

        json_path = f"{self.folder_path}/inputparams.json"
        with open(json_path, "w") as fp:
            json.dump(jsonparams, fp, indent=4)


# ! Simulation parameters
if __name__ == "__main__":

    si = shamrock.UnitSystem()
    sicte = shamrock.Constants(si)

    code_unit = shamrock.UnitSystem(
        unit_length=sicte.earth_radius(),
        unit_mass=sicte.earth_mass(),
        unit_time=np.sqrt(sicte.earth_radius() ** 3.0 / sicte.G() / sicte.earth_mass()),
    )

    rho_center = 1.5 * Tillotson_parameters_Fe["rho0"]
    eos = EoS(id="tillotson", details={"material": "Fe"}, unit=code_unit)
    ball1 = Tillotson_Ball(
        center=[0, 0, 0],
        v_xyz=[1, 0, 0],
        N_target=2e3,
        rho_center=rho_center,
        u_int=0,
        eos=eos,
    )

    #!####################################

    nb_dumps = 5
    tf_cl = 10  # durée de la run en temps de chute libre (environ)
    #! #####################################
    setup = Setup(
        SG="mm", eos=eos, balls=[ball1], nb_dumps=nb_dumps, tf=tf_cl, overwrite=True
    )

    tcl = setup.get_free_fall_time()
    tf = tf_cl * tcl
    t_stops = np.linspace(0, tf, nb_dumps)
    setup.ready_set_go(rescale=False)
    print(t_stops)
    setup.loop(t_stops)

# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./balls.py
