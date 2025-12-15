import shamrock
import numpy as np
import os
import glob
import sham_utilities
import stretchmap_utilities as su
import px_utilities
import ffmpeg


shamrock.enable_experimental_features()


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


def dump(model, dump_path):
    model.dump(dump_path)
    print(f"Dumped {dump_path}")


def plot(fig, img_path, model, ctx, rhotarget, inputparams, eos=None):
    mpart = model.get_particle_mass()

    inputparams["pmass"] = mpart

    if fig is None:
        fig = px_utilities.px_3d_and_rho(model, ctx, img_path, rhotarget, inputparams)
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
    fig = None
    newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
    dump_path = f"{newpath_withoutext}.sham"
    img_path = f"{newpath_withoutext}.png"
    dump(model, dump_path=dump_path)  # **before** plotting
    # fig = plot(None, img_path, model, ctx, rhotarget, inputparams, eos)
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
    # plot(fig, img_path, model, ctx, rhotarget, inputparams)
    dump(model, dump_path=dump_path)
    # fig.show()
    return inputparams


def loop(fig, t_stop, model, ctx, rhotarget, inputparams, dump_prefix):
    for i, t in enumerate(t_stop):
        print(f"looping, still {len(t_stop)-1} to go")
        model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
        model.evolve_until(t)
        model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))
        newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        img_path = f"{newpath_withoutext}.png"
        dump(model, dump_path=dump_path)  # **before** plotting
        # plot(fig, img_path, model, ctx, rhotarget, inputparams)


def setup_Fermi(y0, mu_e):
    tabx, tabrho = su.solve_Chandrasekhar(y0, mu_e)
    tabx /= su.Rsol
    tabrho /= su.density

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
    folder_restart = "./outputs/f2_2000k_SG_cd10_000/"
    overwrite = True
    # durationrestart = 1 #  + 1 fois la simu initiale
    # durationrestart = 0
    SG = "fmm"
    nb_dumps = 200
    tf_cl = 24  # durée de la run en temps de chute libre (environ)

    N_target = 2e5

    eos = "fermi"
    # eos = "polytropic"

    #! #####################################
    inputparams = {}

    if eos == "fermi":
        y0 = 5
        mu_e = 2
        eos = {"name": "fermi", "id": f"f{mu_e}", "values": {"mu_e": mu_e, "y0": y0}}
        tabx, tabrho, mtot_target = setup_Fermi(y0, mu_e)  # already normalized
        # mtot_target *= 1.01
        rhoprofiletxt = "solve_ivp(RK45)"
    elif eos == "polytropic":
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
        # Normalizing
        integral_profile = su.integrate_target([tabx, tabrho])
        rho0 = mtot_target / integral_profile
        rhotarget = np.array([tabx, rho0 * tabrho])

    xmax = np.max(tabx)
    rhotarget = np.array([tabx, tabrho])

    print("max density", np.max(tabrho))
    print("mean density", np.mean(tabrho))
    print("radius", np.max(tabx))
    print("mtot integrated", mtot_target)
    hfact = 1.2  # ou la valeur voulue
    m = mtot_target / (N_target / 2)  # masse par particule
    h = hfact * (m / np.max(tabrho)) ** (1.0 / 3.0)  # h min à peu près
    eps_plummer = h

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
    # model.set_particle_mass(mtot / Npartfinal)
    # print(f"mpart should be {mtot / Npartfinal}")
    # print(f"curretly it is {model.get_particle_mass()}")
    # print(
    #     f"Ended up with {Npartfinal} particles so Mtot={Npartfinal*model.get_particle_mass()}, testing init"
    # )

    if restart:
        # fig = None
        loop(None, t_stop, model, ctx, rhotarget, inputparams, dump_prefix)
        print("Running completed, showing final plot")
        ## ! Video
        # fps = px_utilities.compute_fps(inputparams)
        # pattern_png = f"{folder_path}/*.png"
        # filemp4 = f"{folder_path}/{dump_prefix}.mp4"
        # px_utilities.movie(pattern_png, filemp4, fps)
        # print(f"movie: {filemp4}")
    else:
        ## ! Making sure everything nicely settled
        inputparams = test_init(model, ctx, rhotarget, inputparams, dump_prefix)
        write_json_params(inputparams, json_path=f"{folder_path}/inputparams.json")
        print("Init test completed, running")
        ## ! Running
        loop(None, t_stop, model, ctx, rhotarget, inputparams, dump_prefix)
        print("Running completed, showing final plot")
        ## ! Video
        # fps = px_utilities.compute_fps(inputparams)
        # # fps = 2
        # pattern_png = f"{folder_path}/*.png"
        # filemp4 = f"{folder_path}/{dump_prefix}.mp4"
        # px_utilities.movie(pattern_png, filemp4, fps)
        # print(f"movie: {filemp4}")
        # dump(model, dump_path=f"{folder_name}/finaldump.sham")


# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./test_fermi.py
