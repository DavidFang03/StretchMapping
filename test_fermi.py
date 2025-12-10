import shamrock
import numpy as np
import os
import glob
import sham_utilities
import stretchmap_utilities as su
import px_utilities
import ffmpeg


shamrock.enable_experimental_features()


def handle_dump(dump_prefix):
    try:
        os.mkdir("outputs")
    except OSError as error:
        print("ok outputs exist")
    folder_name = sham_utilities.get_new_folder(dump_prefix)
    try:
        os.mkdir(folder_name)
    except OSError as error:
        print(f"{folder_name} Directory already exists, no need to mkdir a new one.")

    command = f"cp {os.path.abspath(__file__)} {folder_name}/{dump_prefix}.py"
    print("executing", command)
    os.system(command)
    print("Ok let's go with", folder_name)
    return folder_name


def reloadModel(dump_prefix):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
    dump_path = sham_utilities.get_last_dump_path(dump_prefix)
    print("loading from", dump_path)
    model.load_from_dump(dump_path)
    return model, ctx


def adim_r(r, codeu):
    """
    for n=2

    :param r: Description
    """
    ucte = shamrock.Constants(codeu)
    G = ucte.G()
    return r * np.sqrt(4 * np.pi * G / 2)


def get_radius_n1(K, codeu):
    """
    for n = 1

    :param K: Description
    :param codeu: Description
    """
    ucte = shamrock.Constants(codeu)
    G = ucte.G()
    return np.sqrt(np.pi * K / (2 * G))


def initModel():
    if not shamrock.sys.is_initialized():
        shamrock.change_loglevel(1)
        shamrock.sys.init("0:0")

    # -----------------------
    # Code units
    # -----------------------
    # unit of mass = 1 Solar mass
    # unit of length = 1 astronomical unit
    # unit of time is inverse angular frequency (GM/r^3), such that
    # the gravitational constant = 1 in code units
    si = shamrock.UnitSystem()
    sicte = shamrock.Constants(si)
    codeu = shamrock.UnitSystem(
        unit_length=sicte.solar_radius(),
        unit_mass=sicte.sol_mass(),
        unit_time=np.sqrt(sicte.solar_radius() ** 3.0 / sicte.G() / sicte.sol_mass()),
    )

    ucte = shamrock.Constants(codeu)
    G = ucte.G()
    print("G code", G)
    print("G SI", sicte.G())

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    return model, ctx, codeu


def setupModel(model, codeu, dr, xmax, mtot_target, rhotarget, eos, SG, eps_plummer):
    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    cfg.set_particle_tracking(True)  # ! important ?

    if SG:
        cfg.set_self_gravity_mm(
            order=5, opening_angle=0.5, reduction_level=3
        )  #! self-gravity
        cfg.set_softening_plummer(epsilon=eps_plummer)

    if eos["name"] == "fermi":
        cfg.set_eos_fermi(eos["values"]["mu_e"])  # mu_e = 2 e.g
    elif eos["name"] == "polytropic":
        cfg.set_eos_polytropic(eos["values"]["K"], eos["values"]["gamma"])

    cfg.set_units(codeu)
    model.set_solver_config(cfg)
    # model.set_particle_mass(pmass)

    # should be number of SPH particles per GPU / 4?
    # seems that it can be quite large...
    model.init_scheduler(int(1e7), 1)

    # resize simulation box
    sbmin = (-xmax * 2, -xmax * 2, -xmax * 2)
    sbmax = (xmax * 2, xmax * 2, xmax * 2)
    model.resize_simulation_box(sbmin, sbmax)
    bmin = (-xmax, -xmax, -xmax)
    bmax = (xmax, xmax, xmax)

    # generate model setup
    setup = model.get_setup()
    hcp = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    tabx = rhotarget[0]
    tabrho = rhotarget[1]
    stretched_hcp = setup.make_modifier_stretch_mapping(
        parent=hcp,
        system="spherical",
        axis="r",
        box_min=bmin,
        box_max=bmax,
        tabx=tabx,
        tabrho=tabrho,
        mtot=mtot_target,
    )
    setup.apply_setup(stretched_hcp)

    C_cour = 0.3
    C_force = 0.25
    model.set_cfl_cour(C_cour)
    model.set_cfl_force(C_force)
    return model, ctx


def dump(model, dump_path):
    model.dump(dump_path)
    print(f"Dumped {dump_path}")


def plot(fig, img_path, model, ctx, rhotarget, inputparams, eos=None):
    mpart = model.get_particle_mass()

    inputparams["pmass"] = mpart

    if fig is None:
        fig = px_utilities.px_3d_and_rho(
            model, ctx, img_path, rhotarget, inputparams, eos
        )
    else:
        px_utilities.update_px_3d_and_rho(fig, model, ctx, img_path, inputparams)
    print("I will write this image in", img_path)

    fig.write_image(img_path)

    return fig


def write_json_params(inputparams, json_path):
    import json

    with open(json_path, "w") as fp:
        json.dump(inputparams, fp, indent=4)


def test_init(model, ctx, rhotarget, inputparams, dump_prefix):
    """
    Dump and plot initial configuration then show it.
    Evolve with dt = 0, dump and plot.

    :param model: Description
    :param ctx: Description
    :param rhotarget: Description
    :param img_path: Description
    :param dump_path: Description
    """

    newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
    dump_path = f"{newpath_withoutext}.sham"
    img_path = f"{newpath_withoutext}.png"
    dump(model, dump_path=dump_path)  # **before** plotting
    fig = plot(None, img_path, model, ctx, rhotarget, inputparams, eos)
    # fig.show()
    model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
    model.evolve_once_override_time(0.0, 0.0)
    model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))
    newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
    img_path = f"{newpath_withoutext}.png"
    dump_path = f"{newpath_withoutext}.sham"
    plot(fig, img_path, model, ctx, rhotarget, inputparams)
    dump(model, dump_path=dump_path)
    return fig


def loop(fig, t_stop, model, ctx, rhotarget, inputparams, dump_prefix):
    for t in t_stop:
        model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
        model.evolve_until(t)
        model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))
        newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        img_path = f"{newpath_withoutext}.png"
        dump(model, dump_path=dump_path)  # **before** plotting
        plot(fig, img_path, model, ctx, rhotarget, inputparams)


# ! Simulation parameters
if __name__ == "__main__":
    model, ctx, codeu = initModel()

    #####################################
    restart = False
    SG = True
    nb_dumps = 90

    N_target = 2000

    eos = "fermi"
    # eos = "polytropic"

    ######################################
    inputparams = {}

    if eos == "fermi":
        # Then, the input is y0
        y0 = 1.5
        mu_e = 2
        tabx, tabrho = su.solve_Chandrasekhar(y0)
        tabx /= su.Rsol
        tabrho /= su.density
        arr1inds = tabx.argsort()
        tabx = tabx[arr1inds]
        tabrho = tabrho[arr1inds]
        xmax = np.max(tabx)
        rhoprofiletxt = "solve_ivp(RK45)"
        rhotarget = np.array([tabx, tabrho])
        mtot_target = su.integrate_target(rhotarget)
        eps_plummer = np.pow(
            (mtot_target / (N_target / 2)) / np.max(tabrho), 1.0 / 3.0
        ) ** 3 / (
            N_target / 2
        )  # h min à peu près
        eos = {"name": "fermi", "id": f"f{mu_e}", "values": {"mu_e": mu_e}}
        inputparams["y0"] = y0
    elif eos == "polytropic":
        # Then, the input is mtot_target
        eps_plummer = 1e-2
        mtot_target = 3
        K = 1
        n = 1
        eos = {
            "name": "polytropic",
            "id": f"n{n}",
            "values": {"n": n, "gamma": 1 + 1 / n, "K": K},
        }
        xmax = get_radius_n1(eos["values"]["K"], codeu)
        rhoprofile = lambda r: np.sinc(r / xmax)
        rhoprofiletxt = "sinc"
        tabx = np.linspace(0, xmax)
        tabrho = rhoprofile(tabx)
        rhotarget = np.array([tabx, tabrho])

    dump_prefix = f"{eos["id"]}_"

    part_vol = ((2 * xmax) ** 3) / N_target
    HCP_PACKING_DENSITY = 0.74
    part_vol_lattice = HCP_PACKING_DENSITY * part_vol
    dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)
    pmass = mtot_target / N_target
    print(f"Guessing {N_target} particles")
    dump_prefix += f"{int(N_target/1000)}k_"

    if SG:
        dump_prefix += "SG_"
    dump_prefix += "cd10_"

    tf = 2 * xmax ** (1.5)
    # tf = 1
    # tf = 1e-40
    # nb_dumps = 4
    print(f"xmax{xmax:.1e} tf{tf:.1e}")

    t_stop = np.linspace(0, tf, nb_dumps)

    inputparams["nb_dumps"] = nb_dumps
    inputparams["tf"] = tf
    inputparams["pmass"] = f"{pmass:.1e}"  # TODO not anymore..
    inputparams["mtot_target"] = mtot_target  # TODO not anymore..
    inputparams["dr"] = dr
    inputparams["xmax"] = xmax
    inputparams["eos"] = eos["name"]
    inputparams["target"] = rhoprofiletxt
    inputparams["SG"] = SG
    inputparams["eps_plummer"] = eps_plummer

    for param, value in eos["values"].items():
        inputparams[param] = value

    ## ! Set the scene<
    folder_path = handle_dump(dump_prefix)
    write_json_params(inputparams, json_path=f"{folder_path}/inputparams.json")

    ## ! Stretchmapping

    if restart:
        model, ctx = reloadModel(dump_prefix)
    else:
        model, ctx = setupModel(
            model,
            codeu,
            dr,
            xmax,
            mtot_target=mtot_target,
            rhotarget=rhotarget,
            eos=eos,
            SG=SG,
            eps_plummer=eps_plummer,
        )
    Npartfinal = model.get_total_part_count()
    pmassfinal = model.get_particle_mass()
    print(
        f"Ended up with {Npartfinal} particles so Mtot={Npartfinal*pmassfinal}, testing init"
    )
    # model.set_particle_mass(mtot / Npartfinal)
    # print(f"mpart should be {mtot / Npartfinal}")
    # print(f"curretly it is {model.get_particle_mass()}")
    # print(
    #     f"Ended up with {Npartfinal} particles so Mtot={Npartfinal*model.get_particle_mass()}, testing init"
    # )

    ## ! Making sure everything nicely settled
    fig = test_init(model, ctx, rhotarget, inputparams, dump_prefix)
    print("Init test completed, running")
    ## ! Running
    loop(fig, t_stop, model, ctx, rhotarget, inputparams, dump_prefix)
    print("Running completed, showing final plot")
    ## ! Video
    fps = px_utilities.compute_fps(inputparams)
    fps = 1
    pattern_png = f"{folder_path}/*.png"
    filemp4 = f"{folder_path}/{dump_prefix}.mp4"
    px_utilities.movie(pattern_png, filemp4, fps)


# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./test_fermi.py
