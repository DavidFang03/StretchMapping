import shamrock

import numpy as np
import os
import glob
import sham_utilities
import stretchmap_utilities as su
import hydrostatic as hy
import px_utilities
import ffmpeg

shamrock.enable_experimental_features()
do_px = True


def handle_dump(dump_prefix, overwrite=False):
    try:
        os.mkdir("outputs")
    except OSError as error:
        print("ok outputs exist")

    last_eventual_folder_name = sham_utilities.get_last_folder(dump_prefix)
    if overwrite and os.path.isdir(last_eventual_folder_name):
        folder_name = sham_utilities.get_last_folder(dump_prefix)
        user_agree = input(
            f"Will remove the entire {folder_name} folder ({len(glob.glob(f"{folder_name}/*.sham"))} dumps) (y/n)"
        )
        if user_agree == "y":

            for f in glob.glob(f"{folder_name}/*"):
                os.remove(f)
    else:
        folder_name = sham_utilities.get_new_folder(dump_prefix)
        try:
            os.mkdir(folder_name)
        except OSError as error:
            print(
                f"{folder_name} Directory already exists, no need to mkdir a new one."
            )

    command = f"cp {os.path.abspath(__file__)} {folder_name}/{dump_prefix}.py"
    print("executing", command)
    os.system(command)
    print("Ok let's go with", folder_name)
    return folder_name


def reloadModel(dump_prefix, model):
    dump_path = sham_utilities.get_last_dump_path(dump_prefix)
    print("loading from", dump_path)
    model.load_from_dump(dump_path)
    return model


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
    # codeu = shamrock.UnitSystem(
    #     unit_length=sicte.solar_radius(),
    #     unit_mass=sicte.sol_mass(),
    #     unit_time=np.sqrt(sicte.solar_radius() ** 3.0 / sicte.G() / sicte.sol_mass()),
    # )

    codeu = shamrock.UnitSystem(
        unit_length=sicte.earth_radius(),
        unit_mass=sicte.earth_mass(),
        unit_time=np.sqrt(sicte.earth_radius() ** 3.0 / sicte.G() / sicte.earth_mass()),
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

    if SG == "mm":
        cfg.set_self_gravity_mm(
            order=5, opening_angle=0.5, reduction_level=3
        )  #! self-gravity
        cfg.set_softening_plummer(epsilon=eps_plummer)
    elif SG == "fmm":
        cfg.set_self_gravity_fmm(
            order=5, opening_angle=0.5, reduction_level=3
        )  #! self-gravity
        cfg.set_softening_plummer(epsilon=eps_plummer)

    if eos["name"] == "fermi":
        cfg.set_eos_fermi(eos["values"]["mu_e"])  # mu_e = 2 e.g
    elif eos["name"] == "polytropic":
        cfg.set_eos_polytropic(eos["values"]["K"], eos["values"]["gamma"])
    elif eos["name"] == "tillotson":
        # print(recover_tillotson_values(eos["values"]))
        cfg.set_eos_tillotson(**recover_tillotson_values(eos["values"]))

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

    C_cour = 0.1
    C_force = 0.1
    # C_cour = 0.3
    # C_force = 0.25
    model.set_cfl_cour(C_cour)
    model.set_cfl_force(C_force)
    return model, ctx


def recover_tillotson_values(values):
    values_to_shamrock = {}
    for key, value in values.items():
        if key in ["rho0", "E0", "A", "B", "a", "b", "alpha", "beta", "u_iv", "u_cv"]:
            values_to_shamrock[key] = value
    return values_to_shamrock


def dump(model, dump_path):
    model.dump(dump_path)
    print(f"Dumped {dump_path}")


def plot(fig, img_path, model, ctx, rhotarget, inputparams, unit, eos=None):
    mpart = model.get_particle_mass()

    inputparams["pmass"] = mpart

    if fig is None:
        fig = px_utilities.px_3d_and_rho(
            model, ctx, img_path, rhotarget, inputparams, unit
        )
    else:
        px_utilities.update_px_3d_and_rho(fig, model, ctx, img_path, inputparams)
    print("I will write this image in", img_path)
    firstimg_path = img_path[:-5] + "0" + img_path[-4:]
    print("You might wanna check", firstimg_path)
    fig.write_image(img_path)

    return fig


def write_json_params(inputparams, json_path):
    import json

    with open(json_path, "w") as fp:
        json.dump(inputparams, fp, indent=4)


def test_init(model, ctx, rhotarget, inputparams, dump_prefix, unit):
    """
    Dump and plot initial configuration then show it.
    Evolve with dt = 0, dump and plot.

    :param model: Description
    :param ctx: Description
    :param rhotarget: Description
    :param img_path: Description
    :param dump_path: Description
    """
    fig = None
    newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
    dump_path = f"{newpath_withoutext}.sham"
    img_path = f"{newpath_withoutext}.png"
    dump(model, dump_path=dump_path)  # **before** plotting
    if do_px:
        fig = plot(None, img_path, model, ctx, rhotarget, inputparams, unit, eos)
    model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
    model.evolve_once_override_time(0.0, 0.0)
    model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))
    # Here we need to rescale the density : mpart -> mpart*rho_{target}/rho_{SPH} = mpart*
    hpart = ctx.collect_data()["hpart"]
    mpart = model.get_particle_mass()
    rhoSPH = mpart * (model.get_hfact() / hpart) ** 3
    rescale = np.max(rhotarget) / np.max(rhoSPH)
    inputparams["rescale"] = rescale
    model.set_particle_mass(mpart * rescale)

    newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
    img_path = f"{newpath_withoutext}.png"
    dump_path = f"{newpath_withoutext}.sham"
    if do_px:
        plot(fig, img_path, model, ctx, rhotarget, inputparams, unit)
    dump(model, dump_path=dump_path)
    # fig.show()
    return inputparams, fig


def loop(fig, t_stop, model, ctx, rhotarget, inputparams, dump_prefix, unit):
    for i, t in enumerate(t_stop):
        print(f"looping, still {len(t_stop)-1} to go")
        model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
        model.evolve_until(t)
        model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))
        newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        img_path = f"{newpath_withoutext}.png"
        dump(model, dump_path=dump_path)  # **before** plotting
        if do_px:
            plot(fig, img_path, model, ctx, rhotarget, inputparams, unit)


def setup_Fermi(y0, mu_e):
    tabx, tabrho = su.solve_Chandrasekhar(y0, mu_e)
    tabx /= su.Rsol
    tabrho /= su.density

    arr1inds = tabx.argsort()
    tabx = tabx[arr1inds]
    tabrho = tabrho[arr1inds]
    mtot_target = su.integrate_target([tabx, tabrho])

    return tabx, tabrho, mtot_target


def setup_Tillotson(till_values):
    tabx, tabrho = hy.solve_hydrostatic(till_values, codeu)

    arr1inds = tabx.argsort()
    tabx = tabx[arr1inds]
    tabrho = tabrho[arr1inds]
    mtot_target = su.integrate_target([tabx, tabrho])

    return tabx, tabrho, mtot_target


# ! Simulation parameters
if __name__ == "__main__":
    model, ctx, codeu = initModel()

    #!####################################
    restart = False
    durationrestart = 0
    overwrite = True
    # durationrestart = 1 #  + 1 fois la simu initiale
    # durationrestart = 0
    SG = "mm"
    nb_dumps = 200
    tf_cl = 2  # durée de la run en temps de chute libre (environ)

    N_target = 2e3

    eos = "tillotson"
    # For Fe [Tillotson 1962]
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
    kwargs_tillotson = hy.adimension(kwargs_tillotson, codeu)
    # eos = "polytropic"

    #! #####################################
    inputparams = {}

    if eos == "tillotson":
        eos = {"name": "tillotson", "id": f"tillotson", "values": kwargs_tillotson}
        tabx, tabrho, mtot_target = setup_Tillotson(kwargs_tillotson)
        rhoprofiletxt = "solve_ivp(RK45)"

    xmax = np.max(tabx)
    rhotarget = np.array([tabx, tabrho])

    print("max density", np.max(tabrho))
    print("mean density", np.mean(tabrho))
    print("radius", np.max(tabx))
    print("mtot integrated", mtot_target)
    hfact = 1.2  # ou la valeur voulue
    m = mtot_target / (N_target / 2)  # masse par particule
    h = hfact * (m / np.max(tabrho)) ** (1.0 / 3.0)
    eps_plummer = h  # h min à peu près

    dump_prefix = f"{eos["id"]}_"

    part_vol = ((2 * xmax) ** 3) / N_target
    HCP_PACKING_DENSITY = 0.74
    part_vol_lattice = HCP_PACKING_DENSITY * part_vol
    dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)
    pmass = mtot_target / N_target
    print(f"Guessing {N_target} particles")
    dump_prefix += f"{int(N_target/1000)}k_"

    if SG != False:
        dump_prefix += f"{SG}_"
    dump_prefix += "cd10_rescaled_"

    tcl = np.sqrt(xmax**3 / mtot_target)
    tf = tf_cl * tcl
    # tf = 1
    # tf = 5e-2
    if restart:
        t_stop = np.linspace(
            0 + durationrestart * tf,
            tf + durationrestart * tf,
            nb_dumps + durationrestart * nb_dumps,
        )
        tf *= 2
        nb_dumps *= 2
    else:
        t_stop = np.linspace(0, tf, nb_dumps)
    print(f"xmax{xmax:.1e} tf{tf:.1e}")

    inputparams["nb_dumps"] = nb_dumps
    inputparams["tf"] = tf
    inputparams["pmass"] = f"{pmass:.1e}"  # TODO not anymore..
    inputparams["mtot_target"] = mtot_target
    inputparams["dr"] = dr
    inputparams["xmax"] = xmax
    inputparams["eos"] = eos
    inputparams["target"] = rhoprofiletxt
    inputparams["SG"] = SG
    inputparams["eps_plummer"] = eps_plummer

    for param, value in eos["values"].items():
        inputparams[param] = value

    ## ! Set the scene
    if restart:
        folder_path = folder_restart
    else:
        folder_path = handle_dump(dump_prefix, overwrite)

    ## ! Stretchmapping

    if restart:
        model = reloadModel(dump_prefix, model)
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
    if restart:
        # fig = None
        loop(None, t_stop, model, ctx, rhotarget, inputparams, dump_prefix, codeu)
        print("Running completed, showing final plot")
        ## ! Video
        # fps = px_utilities.compute_fps(inputparams)
        # pattern_png = f"{folder_path}/*.png"
        # filemp4 = f"{folder_path}/{dump_prefix}.mp4"
        # px_utilities.movie(pattern_png, filemp4, fps)
        # print(f"movie: {filemp4}")
    else:
        ## ! Making sure everything nicely settled
        inputparams, fig = test_init(
            model, ctx, rhotarget, inputparams, dump_prefix, codeu
        )
        write_json_params(inputparams, json_path=f"{folder_path}/inputparams.json")
        print("Init test completed, running")
        ## ! Running
        loop(fig, t_stop, model, ctx, rhotarget, inputparams, dump_prefix, codeu)
        print("Running completed, showing final plot")
        ## ! Video
        if do_px:
            fps = px_utilities.compute_fps(inputparams)
            # fps = 2
            pattern_png = f"{folder_path}/*.png"
            filemp4 = f"{folder_path}/{dump_prefix}.mp4"
            px_utilities.movie(pattern_png, filemp4, fps)
            print(f"movie: {filemp4}")


# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./test_tillotson.py
