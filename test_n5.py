import shamrock
import numpy as np
import os
import glob
import sham_utilities
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
    model.load_from_dump(dump_path)
    return ctx, model


def adim_r(r, codeu):
    """
    for n=2

    :param r: Description
    """
    ucte = shamrock.Constants(codeu)
    G = ucte.G()
    return r * np.sqrt(4 * np.pi * G / 2)


def get_radius(K, codeu):
    """
    for n =1

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
        unit_length=sicte.au(),
        unit_mass=sicte.sol_mass(),
        unit_time=np.sqrt(sicte.au() ** 3.0 / 6.67e-11 / sicte.sol_mass()),
    )

    ucte = shamrock.Constants(codeu)
    G = ucte.G()
    print("G", G)

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    return model, ctx, codeu


def setupModel(model, codeu, dr, xmax, pmass, rhoprofile, K, gamma, SG):

    # ? Integrator parameters (parameters in CFL condition)
    C_cour = 0.3
    C_force = 0.25
    cfg = model.gen_default_config()

    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    # ? set_artif_viscosity_Constant(typename AVConfig::Constant v) vs set_artif_viscosity_ConstantDisc(typename AVConfig::ConstantDisc v) ?
    cfg.set_particle_tracking(True)  # ! important ?

    if SG:
        cfg.set_self_gravity_mm(
            order=5, opening_angle=0.5, reduction_level=3
        )  #! self-gravity
        cfg.set_softening_plummer(epsilon=1e-9)

    cfg.set_eos_polytropic(K, gamma)  # n = 1
    # cfg.print_status()
    cfg.set_units(codeu)
    model.set_solver_config(cfg)
    model.set_particle_mass(pmass)

    # should be number of SPH particles per GPU / 4?
    # seems that it can be quite large...
    model.init_scheduler(int(1e7), 1)

    # resize simulation box
    sbmin = (-xmax * 2, -xmax * 2, -xmax * 2)
    sbmax = (xmax * 2, xmax * 2, xmax * 2)
    model.resize_simulation_box(sbmin, sbmax)
    bmin = (-xmax, -xmax, -xmax)
    bmax = (xmax, xmax, xmax)
    # center = (0,0,0)

    # generate model setup
    setup = model.get_setup()

    # gen = setup.make_generator_lattice_hcp_smap(dr, bmin, bmax, [rhoprofile], "spherical", ["r"])
    # setup.apply_setup(gen)
    hcp = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    stretched_hcp = setup.make_modifier_stretch_mapping(
        parent=hcp,
        rhoprofiles=[rhoprofile],
        system="spherical",
        axes=["r"],
        box_min=bmin,
        box_max=bmax,
    )
    setup.apply_setup(stretched_hcp)

    model.set_cfl_cour(C_cour)
    model.set_cfl_force(C_force)

    # convergence for smoothing length
    # model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
    # model.timestep()
    # model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))
    print("hpart: ", ctx.collect_data()["hpart"])
    return model, ctx


def dump(model, dump_path):
    model.dump(dump_path)
    print(f"Dumped {dump_path}")


def plot(model, ctx, rhotarget, intputparams, img_path):
    data = ctx.collect_data()
    mpart = model.get_particle_mass()
    t = model.get_time()
    fig = px_utilities.px_3d_and_rho(data, rhotarget, mpart, t, intputparams, img_path)

    print("I will write this image in", img_path)
    fig.write_image(img_path)

    return fig


def write_json_params(inputparams, json_path):
    import json

    # TODO - To format

    with open(json_path, "w") as fp:
        json.dump(inputparams, fp)


def test_init(model, ctx, rhoprofile, intputparams, dump_prefix):
    """
    Dump and plot initial configuration then show it.
    Evolve with dt = 0, dump and plot.

    :param model: Description
    :param ctx: Description
    :param rhoprofile: Description
    :param img_path: Description
    :param dump_path: Description
    """
    newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)

    newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
    dump_path = f"{newpath_withoutext}.sham"
    img_path = f"{newpath_withoutext}.png"
    dump(model, dump_path=dump_path)  # **before** plotting
    fig = plot(model, ctx, rhoprofile, intputparams, img_path)
    fig.show()
    model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
    model.evolve_once_override_time(0.0, 0.0)
    model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))
    plot(model, ctx, rhoprofile, intputparams, img_path=img_path)
    dump(model, dump_path=dump_path)


def loop(t_stop, model, ctx, rhoprofile, intputparams, dump_prefix):
    for t in t_stop:
        model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
        model.evolve_until(t)
        model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))
        newpath_withoutext = sham_utilities.gen_new_path_withoutext(dump_prefix)
        dump_path = f"{newpath_withoutext}.sham"
        img_path = f"{newpath_withoutext}.png"
        dump(model, dump_path=dump_path)  # **before** plotting
        fig = plot(model, ctx, rhoprofile, intputparams, img_path=img_path)

    return fig


restart = True
# ! Simulation parameters
if __name__ == "__main__":
    model, ctx, codeu = initModel()
    # if restart:
    #     model, ctx, ucode = reloadModel(dump_prefix)
    SG = True
    n = 1
    gamma = 1.0 + (1 / n)
    K = 1
    dump_prefix = f"cd10_n{n}_"

    Npart_i = 50
    xmax = get_radius(K, codeu)
    dr = 2 * xmax / (Npart_i - 1)

    estimated_Npart = (xmax / dr) ** 3 * (xmax**3 / (4 * np.pi * xmax**3 / 3))
    pmass = 1 / estimated_Npart
    dump_prefix += f"1e{int(np.log10(abs(estimated_Npart)))}_"
    if SG:
        dump_prefix += "SG_"

    if n == 5:
        rhoprofile = lambda r: 1 / np.sqrt(1 + (r**2) / 3)
        rhoprofiletxt = "1/np.sqrt(1+(r**2)/3)"
    elif n == 1:
        rhoprofile = lambda r: np.sinc(adim_r(r, codeu) / np.pi)
        rhoprofiletxt = "sinc"
    # rhoprofile = lambda r: 1/np.sqrt(1+r**2)
    # rhoprofile = lambda r: 1/((r+0.1)**2)
    # rhoprofile = lambda r: 1/((r+0.01)**2)
    # rhoprofile = lambda r:
    # rhoprofile = lambda r: np.sinc(r)
    # rhoprofile = np.sinc
    nb_dumps = 180
    tf = 3
    t_stop = np.linspace(0, tf, nb_dumps)

    intputparams = {
        "nb_dumps": nb_dumps,
        "tf": tf,
        "pmass": f"{pmass:.1e}",
        "dr": dr,
        "xmax": xmax,
        "K": K,
        "n": n,
        "target": rhoprofiletxt,
        "SG": SG,
    }

    ## ! Set the scene<
    folder_path = handle_dump(dump_prefix)
    write_json_params(intputparams, json_path=f"{folder_path}/inputparams.json")

    ## ! Stretchmapping
    model, ctx = setupModel(
        model, codeu, dr, xmax, pmass, rhoprofile=rhoprofile, K=K, gamma=gamma, SG=SG
    )
    Npartfinal = model.get_total_part_count()
    print(
        f"Ended up with {Npartfinal} particles so Mtot={Npartfinal*pmass}, testing init"
    )

    ## ! Making sure everything nicely settled
    test_init(model, ctx, rhoprofile, intputparams, dump_prefix)
    print("Init test completed, running")
    ## ! Running
    fig = loop(t_stop, model, ctx, rhoprofile, intputparams, dump_prefix)
    print("Running completed, showing final plot")
    ## ! Final plot
    # for fig in figs:
    fig.show()
    ## ! Video
    fps = px_utilities.compute_fps(intputparams)
    pattern_png = f"{folder_path}/*.png"
    filemp4 = f"{folder_path}/{dump_prefix}.mp4"
    px_utilities.movie(pattern_png, filemp4, fps)


# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./test_n5.py
## ? Warning: the corrector tolerance are broken the step will be re rerunned                                                                                                                                    [BasicGasSPH][rank=0]
# ? eps_v = 0.06158665025247084 ???
