import shamrock

if not shamrock.sys.is_initialized():
    print("coucou")
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

import numpy as np
import sham_utilities as shu
import stretchmap_utilities as stu
import hydrostatic as hy
import unitsystem
import matplotlib.pyplot as plt

# import vtkmodules.vtkRenderingFreeType
# import vtkmodules.vtkRenderingMatplotlib
import pyvista as pv
import gc

from rich.console import Console

console = Console()

color_text = "black"

pv.set_plot_theme("dark")
color_text = "white"
pv.global_theme.font.color = color_text


shamrock.enable_experimental_features()

style_target = "dashed"
# colors = {
# "density": "blue",
# "energy": "orange",
# "pressure": "green",
# "soundspeed": "magenta",
# "cold_energy": "red",
# }
colors = {
    "density": "#1e90ff",
    "energy": "#ff7f50",
    "pressure": "#7bed9f",
    "soundspeed": "#5352ed",
    "cold_energy": "#eccc68",
}
# fmts = {
#     "density": "blue--",
#     "energy": "orange--",
#     "pressure": "g--",
#     "soundspeed": "m--",
#     "cold_energy": "r--",
# }
fmts = {
    "density": f"{colors['density']}--",
    "energy": f"{colors['energy']}--",
    "pressure": f"{colors['pressure']}--",
    "soundspeed": f"{colors['soundspeed']}--",
    "cold_energy": f"{colors['cold_energy']}--",
}

ids = {
    "density": "rho",
    "energy": "uint",
    "pressure": "pressure",
    "soundspeed": "soundspeed",
    "cold_energy": "u_c",
}
symbols = {
    "density": r"$\rho$",
    "energy": r"$u$",
    "pressure": r"$p$",
    "soundspeed": r"$c_s$",
    "cold_energy": r"$u_c$",
}


Tillotson_parameters_Fe = hy.Tillotson_parameters_Fe

Tillotson_parameters_Granite = hy.Tillotson_parameters_Granite


class ShamPlot:
    def __init__(self, model, ctx, balls=[]):
        self.ctx = ctx
        self.model = model
        self.balls = balls

        self.subplot_nbs = {
            "density": 0,
            "energy": 1,
            "soundspeed": 2,
            "pressure": 3,
            "cold_energy": 4,
            "mesh": 5,
        }

        self.graphs = {
            "density": None,
            "energy": None,
            "soundspeed": None,
            "pressure": None,
            "cold_energy": None,
        }

        self.charts = {
            "density": None,
            "energy": None,
            "soundspeed": None,
            "pressure": None,
            "cold_energy": None,
        }

        self.first_pvplot()

    def first_pvplot(self):
        self.update_data()
        self.plotter = pv.Plotter(
            shape="5|1",
            splitting_position=0.4375,
            window_size=(960, 540),
            off_screen=True,
        )

        for i, quantity in enumerate(
            ["density", "energy", "soundspeed", "pressure", "cold_energy"]
        ):
            id = ids[quantity]
            pv_color = pv.Color(colors[quantity], opacity=0.5)
            self.plotter.subplot(i)
            chart = pv.Chart2D()
            self.graphs[quantity] = chart.scatter(
                self.data_sham["r"], self.data_sham[id], color=pv_color, size=5
            )
            if len(self.balls) > 0:
                chart.plot(
                    self.balls[0].data["r"],
                    self.balls[0].data[quantity],
                    fmt=fmts[quantity],
                )
            self.plotter.add_chart(chart)
            if i < 4:
                chart.x_axis.toggle()
                chart.x_axis.ticks_visible = False
                chart.x_axis.tick_labels_visible = False
                chart.x_label = ""
            else:
                chart.x_label = r"$r$"
            chart.title = quantity
            chart.y_label = id

            chart.background_color = "#2f3542"
            chart.border_color = "#57606f"

            self.charts[quantity] = chart

        self.plotter.subplot(i + 1)
        point_cloud = pv.PolyData(self.data_sham["xyz"])
        point_cloud[r"\rho"] = self.data_sham["rho"]
        self.mesh_actor = self.plotter.add_mesh(
            point_cloud,
            cmap="cool",
            render_points_as_spheres=True,
            point_size=5.0,
        )
        self.plotter.show_bounds(
            bounds=[-3, 3, -3, 3, -3, 3],
            grid="back",
            location="outer",
            ticks="both",
            n_xlabels=2,
            n_ylabels=2,
            n_zlabels=2,
        )
        self.plotter.camera.position = (-10, 10, 10)
        self.update_time()

    def update_data(self):
        data_sham = self.ctx.collect_data()
        data_sham["r"] = np.linalg.norm(data_sham["xyz"], axis=1)
        data_sham["rho"] = (
            self.model.get_particle_mass()
            * (self.model.get_hfact() / data_sham["hpart"]) ** 3
        )
        data_sham["u_c"] = hy.get_tillotson_cold_energy_granite(data_sham["rho"])

        self.data_sham = data_sham

    def update_time(self):
        self.plotter.subplot(self.subplot_nbs["mesh"])
        if "timetext" in self.plotter.actors:
            self.plotter.remove_actor("timetext")
        self.plotter.add_text(
            "t = {:.1e} dt = {:.1e}".format(self.model.get_time(), self.model.get_dt()),
            position="upper_edge",
            name="timetext",
        )

    def update_pvplot(self):
        self.update_data()

        gc.collect()
        # self.chart.clear()
        for i, quantity in enumerate(
            ["density", "energy", "soundspeed", "pressure", "cold_energy"]
        ):
            self.plotter.subplot(i)
            chart = self.charts[quantity]
            id = ids[quantity]
            chart.clear()
            pv_color = pv.Color(colors[quantity], opacity=0.5)
            self.graphs[quantity] = chart.scatter(
                self.data_sham["r"], self.data_sham[id], color=pv_color, size=5
            )
            # self.graphs[quantity].update(data["r"], data[ids[quantity]])
            # self.charts[quantity].toggle()

        self.plotter.subplot(i + 1)
        new_cloud = pv.PolyData(self.data_sham["xyz"])
        new_cloud[r"$\\rho$"] = self.data_sham["rho"]
        self.mesh_actor.mapper.SetInputData(new_cloud)

        self.update_time()

        self.plotter.render()
        # self.plotter.update()

    def screenshot(self, img_path):
        self.plotter.screenshot(img_path, scale=2)


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
        self.material = self.details.get("material", "Granite")

        match self.material:
            case "Fe":
                params = Tillotson_parameters_Fe
            case "Granite":
                params = Tillotson_parameters_Granite
        self.originalparams = params.copy()
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
            return formated_dict

        eos_json = {
            "id": self.id,
            "params": format_dict(self.params),
            "details": format_dict(self.details),
        }
        if self.id == "tillotson":
            eos_json["material"] = self.material

        return eos_json

    def ask(self, key):
        if key not in self.originalparams:
            raise Exception()
        return self.originalparams[key]


class Tillotson_Ball:
    def __init__(self, center, v_xyz, N_target, rho_center, u_int, eos, rescale=1.0):
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
        self.center = np.array(center)
        self.v_xyz = (v_xyz[0], v_xyz[1], v_xyz[2])
        self.N_target = int(N_target)
        self.rescale = rescale

        rho_center = rho_center / unitsystem.density(eos.unit)
        u_int = u_int / unitsystem.energy(eos.unit)

        self.data = {
            "r": None,
            "density": None,
            "energy": None,
            "pressure": None,
            "soundspeed": None,
            "cold_energy": None,
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

        self.data["cold_energy"] = hy.get_tillotson_cold_energy_granite(
            self.data["density"]
        )

        # self.show_profile()  # DEBUG

    def show_profile(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.data["r"], self.data["density"], marker="x")
        ax.set_xlabel("$r$")
        ax.set_ylabel("$\\rho$")
        plt.show()
        exit()


class Setup:
    def __init__(self, SG, eos, balls, nb_dumps, tf, clear):
        self.SG = SG
        self.eos = eos
        self.balls = balls
        self.unit = eos.unit
        self.ready_toloop = False
        self.fig = None
        self.nb_dumps = nb_dumps
        self.tf = tf

        self.ctx = shamrock.Context()
        self.ctx.pdata_layout_new()
        self.model = shamrock.get_Model_SPH(
            context=self.ctx, vector_type="f64_3", sph_kernel="M4"
        )
        self.cfg = self.model.gen_default_config()

        self.eps_plummer = self.get_eps_plummer()

        self.dump_prefix = self.gen_dump_prefix()
        self.folder_path = shu.handle_dump(
            __file__, dump_prefix=self.dump_prefix, clear=clear
        )
        self.write_json_params()

        self.init_model()

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

        bsize = 10
        sbmin = (-bsize, -bsize, -bsize)
        sbmax = (bsize, bsize, bsize)
        self.cfg.add_kill_sphere(
            center=(0, 0, 0), radius=bsize
        )  # kill particles outside the simulation box

        # -------------------------------------------------------------------------------
        # units  ------------------------------------------------------------------------
        self.cfg.set_units(self.unit)
        # _______________________________________________________________________________

        self.model.set_solver_config(self.cfg)
        self.model.init_scheduler(int(1e7), 1)

        # -------------------------------------------------------------------------------
        # resize simulation box ---------------------------------------------------------

        self.model.resize_simulation_box(sbmin, sbmax)
        # TODO Probably will have to add a kill sphere

        # _______________________________________________________________________________

        self.add_balls()

        # -------------------------------------------------------------------------------
        # CFL ---------------------------------------------------------------------------
        C_cour = 0.1
        C_force = 0.1
        self.model.set_cfl_cour(C_cour)
        self.model.set_cfl_force(C_force)
        # _______________________________________________________________________________

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
                mtot=ball.mtot_target * ball.rescale,
            )

            offset_hcp = setup.make_modifier_offset(
                parent=stretched_hcp,
                offset_position=(0, 0, 0),
                offset_velocity=ball.v_xyz,
            )

            setup.apply_setup(offset_hcp)

    def gen_dump_prefix(self):
        balls_nb = len(self.balls)
        dump_prefix = f"1ball_" if balls_nb == 1 else f"{balls_nb}balls_"
        dump_prefix += f"{self.eos.id}_"
        dump_prefix += f"{int(self.balls[0].N_target/1000)}k_"
        dump_prefix += f"{self.SG}_"
        dump_prefix += "cd10_uc_skewv3_"
        return dump_prefix

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
        console.print(f"Dumped {dump_path}")

        #     mesh_actor.mapper.SetInputData(new_cloud) pour update

    def ready_set_go(self):
        """
        Roadmap:
            - First dump #//(and first plot?)
            - One timestep with dt=0
            - # ?Rescale stretchmapping? Idk how to do it if there are multiple balls.
        """
        self.plot = ShamPlot(self.model, self.ctx, self.balls)

        newpath_withoutext = shu.gen_new_path_withoutext(self.dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        img_path = f"{newpath_withoutext}.png"
        self.dump(dump_path)
        self.plot.screenshot(img_path)

        self.model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
        self.model.evolve_once_override_time(0.0, 0.0)
        self.model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))

        newpath_withoutext = shu.gen_new_path_withoutext(self.dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        img_path = f"{newpath_withoutext}.png"
        self.dump(dump_path)
        self.plot.screenshot(img_path)

        self.ready_toloop = True
        console.print("Dump success: ready to smash balls")

    def loop(self, t_stops):
        if not self.ready_toloop:
            raise Exception("Setup is not ready")
        for i, t in enumerate(t_stops):
            console.print(f"looping {i}, still {len(t_stops)-i} to go")
            self.model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
            self.model.evolve_until(t)
            self.model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))

            newpath_withoutext = shu.gen_new_path_withoutext(self.dump_prefix)
            dump_path = f"{newpath_withoutext}.sham"
            img_path = f"{newpath_withoutext}.png"
            self.dump(dump_path=dump_path)  # **BEFORE** plotting
            self.plot.update_pvplot()
            self.plot.screenshot(img_path)

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

    def movie(self):
        import ffmpeg

        fps = self.nb_dumps / 3
        pattern_png = f"{self.folder_path}/*.png"
        filemp4 = f"{self.folder_path}/{self.dump_prefix}.mp4"
        ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).output(
            filemp4,
            vcodec="libx264",
            crf=18,
            preset="medium",
            r=fps,
            pix_fmt="yuv420p",
            movflags="faststart",
        ).overwrite_output().run()
        console.print(f"movie: {filemp4}")


# ! Simulation parameters
if __name__ == "__main__":

    si = shamrock.UnitSystem()
    sicte = shamrock.Constants(si)

    code_unit = shamrock.UnitSystem(
        unit_length=sicte.earth_radius(),
        unit_mass=sicte.earth_mass(),
        unit_time=np.sqrt(sicte.earth_radius() ** 3.0 / sicte.G() / sicte.earth_mass()),
    )

    eos = EoS(id="tillotson", details={"material": "Granite"}, unit=code_unit)
    rho0 = eos.ask("rho0")
    balls = []
    proto_earth = Tillotson_Ball(
        center=[2, 0, 0],
        v_xyz=[-0.2, 0, 0],
        N_target=2e5,
        rho_center=5.0 * rho0,
        u_int=0.0,
        eos=eos,
        rescale=1.05,
    )

    balls.append(proto_earth)

    theia = Tillotson_Ball(
        center=[-2, 0, 0],
        v_xyz=[0.4, 0.3, 0],
        N_target=1e5,
        rho_center=2 * rho0,
        u_int=0.0,
        eos=eos,
        rescale=1.05,
    )

    balls.append(theia)

    #!####################################

    nb_dumps = 600
    tf_cl = 18  # durée de la run en temps de chute libre (environ)
    #! #####################################
    setup = Setup(
        SG="mm", eos=eos, balls=balls, nb_dumps=nb_dumps, tf=tf_cl, clear=True
    )

    tcl = setup.get_free_fall_time()
    tf = tf_cl * tcl
    t_stops = np.linspace(0, tf, nb_dumps)
    setup.ready_set_go()
    console.print(t_stops)
    setup.loop(t_stops)
    setup.movie()

# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./balls.py
