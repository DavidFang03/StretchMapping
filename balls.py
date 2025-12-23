import numpy as np
import sham_utilities as su
import hydrostatic as hy
import unitsystem

import shamrock


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
        | values=...Stretchmapping parameters (rhocenter here))
    details must contain
        | "material" : Fe e.g
    """

    def __init__(self, id, unit, details={}):
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
        self.values = values
        if self.id == "tillotson":
            rho_center = values["rho_center"]
            u_int = values["u_int"]
            tabx, tabrho = hy.solve_hydrostatic(
                self.params, rho_center, u_int, self.unit
            )
        return tabx, tabrho


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
        self.v_xyz = center
        self.N_target = center

        rho_center = rho_center / unitsystem.density(eos.unit)
        u_int = u_int / unitsystem.energy(eos.unit)

        (
            self.tabx,
            self.tabrho,
        ) = eos.solve_hydrostatic({"rho_center": rho_center, u_int: "u_int"})
        self.mtot_target = su.integrate_target([self.tabx, self.tabrho])

        xmax = np.max(self.tabx)
        part_vol = ((2 * xmax) ** 3) / N_target
        HCP_PACKING_DENSITY = 0.74
        part_vol_lattice = HCP_PACKING_DENSITY * part_vol
        self.dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)


class Setup:
    def __init__(self, SG, eos):
        self.SG = SG
        self.eos = eos
        self.unit = eos.unit
        self.eps_plummer = 1e4
        self.model, self.ctx, self.codeu = self.init_model()

    def init_model(self):
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

        ctx = shamrock.Context()
        ctx.pdata_layout_new()
        model = shamrock.get_Model_SPH(
            context=ctx, vector_type="f64_3", sph_kernel="M4"
        )
        cfg = model.gen_default_config()

        # -------------------------------------------------------------------------------
        # units  ------------------------------------------------------------------------
        cfg.set_units(self.unit)
        # _______________________________________________________________________________

        # -------------------------------------------------------------------------------
        # artificial viscocity  ----------------------------------------------------------
        cfg.set_artif_viscosity_VaryingCD10(
            alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
        )
        # _______________________________________________________________________________

        self.add_balls()

        # -------------------------------------------------------------------------------
        # Self-gravity ------------------------------------------------------------------
        if self.SG == "mm":
            cfg.set_self_gravity_mm(
                order=5, opening_angle=0.5, reduction_level=3
            )  #! self-gravity
        elif self.SG == "fmm":
            cfg.set_self_gravity_fmm(
                order=5, opening_angle=0.5, reduction_level=3
            )  #! self-gravity
        cfg.set_softening_plummer(epsilon=self.eps_plummer)

        if self.eos["name"] == "fermi":
            cfg.set_eos_fermi(self.eos["values"]["mu_e"])
        elif self.eos["name"] == "polytropic":
            cfg.set_eos_polytropic(self.eos["values"]["K"], self.eos["values"]["gamma"])
        elif self.eos["name"] == "tillotson":
            cfg.set_eos_tillotson(**hy.recover_tillotson_values(self.eos["values"]))
        # _______________________________________________________________________________

        model.set_solver_config(cfg)
        model.init_scheduler(int(1e7), 1)

        # -------------------------------------------------------------------------------
        # resize simulation box ---------------------------------------------------------
        sbmin = (-3, -3, -3)
        sbmax = (3, 3, 3)
        model.resize_simulation_box(sbmin, sbmax)
        # _______________________________________________________________________________

        # -------------------------------------------------------------------------------
        # CFL ---------------------------------------------------------------------------
        C_cour = 0.1
        C_force = 0.1
        model.set_cfl_cour(C_cour)
        model.set_cfl_force(C_force)
        # _______________________________________________________________________________

        self.model = model
        self.ctx = ctx

    # def setup_json_params():
    #     self.json_params = {
    #         "nb_dumps": nb_dumps,
    #         "tf": tf,
    #         "pmass": f"{pmass:.1e}",
    #         "mtot_target": self.mtot_target,
    #         "dr": dr,
    #         "xmax": xmax,
    #         "eos": self.eos,
    #         "target": rhoprofiletxt,
    #         "SG": SG,
    #         "eps_plummer": eps_plummer,
    #         "rhoprofiletxt": "solve_ivp(RK45)",
    #     }

    def add_balls(self):
        """
        Also sets self.esp_plummer

        :param center:
        :param v_xyz:
        :param N_target:
        :param eos:
        :param rho_center:
        :param u_int:
        """
        self.eps_plummer = 0
        for ball in self.balls:
            hfact = self.model.get_hfact()  # ou la valeur voulue
            mpart_target = ball.mtot_target / (ball.N_target / 2)  # masse par particule
            hmin = hfact * (mpart_target / np.max(ball.tabrho)) ** (1.0 / 3.0)
            if self.eps_plummer > hmin:
                self.eps_plummer = hmin  # h min à peu près
            # TODO During impact, eps_plummer may have to be smaller ?

            xmax = np.max(ball.tabx)
            ball.bmin = ball.center - np.array([xmax, xmax, xmax])
            ball.bmax = ball.center + np.array([-xmax, -xmax, -xmax])

            setup = self.model.get_setup()
            hcp = setup.make_generator_lattice_hcp(ball.dr, ball.bmin, ball.bmax)
            stretched_hcp = setup.make_modifier_stretch_mapping(
                parent=hcp,
                system="spherical",
                axis="r",
                box_min=ball.bmin,
                box_max=ball.bmax,
                tabx=ball.tabx,
                tabrho=ball.tabrho,
                mtot=ball.mtot_target,
            )
            setup.apply_setup(stretched_hcp)
            self.balls.append(ball)

    def gen_dump_prefix(self):
        balls_nb = len(self.balls)
        dump_prefix = f"1ball_" if balls_nb == 1 else f"{balls_nb}balls_"
        dump_prefix += f"{self.eos["id"]}_"
        dump_prefix += f"{int(self.N_target/1000)}k_"
        dump_prefix += f"{SG}_"
        dump_prefix += "cd10_"

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

    def plot(self):
        pass

    def ready_set_go(self, rescale=False):
        """
        Roadmap:
            - First dump #//(and first plot?)
            - One timestep with dt=0
            - # ?Rescale stretchmapping? Idk how to do it if there are multiple balls.
        """
        newpath_withoutext = su.gen_new_path_withoutext(self.dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        # // fig = None
        # //img_path = f"{newpath_withoutext}.png"
        # //if do_px:
        # //    plot(fig, img_path, model, ctx, rhotarget, inputparams, unit)
        self.dump(dump_path)

        self.model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
        self.model.evolve_once_override_time(0.0, 0.0)
        self.model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))

        newpath_withoutext = su.gen_new_path_withoutext(self.dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        # //img_path = f"{newpath_withoutext}.png"
        # //if do_px:
        # //    plot(fig, img_path, model, ctx, rhotarget, inputparams, unit)
        self.dump(dump_path)

    def loop(self, t_stop):
        for i, t in enumerate(t_stop):
            print(f"looping {i}, still {len(t_stop)-1} to go")
            self.model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
            self.model.evolve_until(t)
            self.model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))

            newpath_withoutext = su.gen_new_path_withoutext(self.dump_prefix)
            dump_path = f"{newpath_withoutext}.sham"
            # //img_path = f"{newpath_withoutext}.png"
            self.dump(dump_path=dump_path)  # **BEFORE** plotting
            # //if do_px:
            # //    plot(fig, img_path, model, ctx, rhotarget, inputparams, unit)


# ! Simulation parameters
if __name__ == "__main__":

    si = shamrock.UnitSystem()
    sicte = shamrock.Constants(si)

    code_unit = shamrock.UnitSystem(
        unit_length=sicte.earth_radius(),
        unit_mass=sicte.earth_mass(),
        unit_time=np.sqrt(sicte.earth_radius() ** 3.0 / sicte.G() / sicte.earth_mass()),
    )

    eos = EoS(id="tillotson", details={"material": "Fe"}, unit=code_unit)
    ball1 = Tillotson_Ball(
        center=[-1, 0, 0],
        v_xyz=[1, 0, 0],
        N_target=2e3,
        rho_center=1.5 * Tillotson_parameters_Fe["rho0"],
        u_int=0,
        eos=eos,
    )

    setup = Setup(SG="mm", eos=eos, balls=[ball1])

    #!####################################
    durationrestart = 0
    overwrite = True
    # durationrestart = 1 #  + 1 fois la simu initiale
    # durationrestart = 0
    SG = "mm"
    nb_dumps = 1000
    tf_cl = 10  # durée de la run en temps de chute libre (environ)

    # eos = "polytropic"

    #! #####################################
    inputparams = {}

    tcl = setup.get_free_fall_time()
    tf = tf_cl * tcl
    t_stop = np.linspace(0, tf, nb_dumps)

    ## ! Set the scene

    folder_path = tt.handle_dump(dump_prefix, overwrite)

    ## ! Stretchmapping

    Npartfinal = model.get_total_part_count()
    pmassfinal = model.get_particle_mass()
    print(
        f"Ended up with {Npartfinal} particles so Mtot={Npartfinal*pmassfinal}, testing init"
    )
    ## ! Making sure everything nicely settled
    inputparams, fig = tt.test_init(
        model, ctx, rhotarget, inputparams, dump_prefix, codeu
    )
    tt.write_json_params(inputparams, json_path=f"{folder_path}/inputparams.json")
    print("Init test completed, running")
    ## ! Running
    tt.loop(fig, t_stop, model, ctx, rhotarget, inputparams, dump_prefix, codeu)
    print("Running completed, showing final plot")


# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./balls.py
