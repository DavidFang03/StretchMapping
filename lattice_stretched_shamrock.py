import shamrock
import numpy as np
import os
import sham_utilities
import px_utilities


shamrock.enable_experimental_features()

def handle_dump(dump_prefix):
    folder_name = sham_utilities.get_folder(dump_prefix)
    try:
        os.mkdir(folder_name)
    except OSError as error:
        print(f"{folder_name} Directory already exists, no need to mkdir a new one.")

    command = f'cp {os.path.abspath(__file__)} {folder_name}/{folder_name}.py'
    os.system(command)
    print("executing",command)

    last_dump_nb = sham_utilities.get_last_dump(dump_prefix)

    return last_dump_nb, folder_name

def Model(dr,xmax,pmass,rhoprofile):
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
        unit_time=np.sqrt( sicte.au()**3.0 / 6.67e-11 / sicte.sol_mass() ),
    )
    ucte = shamrock.Constants(codeu)
    G = ucte.G()
    # M4 -> M6 or M8 would be more precise
    # ? Viscosity parameters ?
    # alpha_ss ~ alpha_AV * 0.08 (ref: Eq. 124 Price+ Phantom paper)
    alpha_ss = 1.0e-4 # (L'instabilité apparaît même avec viscosité) = ?
    alpha_AV = alpha_ss / 0.08
    alpha_u = 1.0
    beta_AV = 2.0

    # ? Integrator parameters (parameters in CFL condition)
    C_cour = 0.3
    C_force = 0.25

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
    cfg = model.gen_default_config()
    # ! Ici on switch les fonctions disques > spherr
    # cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_artif_viscosity_Constant(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    # ? set_artif_viscosity_Constant(typename AVConfig::Constant v) vs set_artif_viscosity_ConstantDisc(typename AVConfig::ConstantDisc v) ?
    # cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)
    cfg.set_particle_tracking(True) # ! important
    cfg.set_eos_locally_isothermal()
    cfg.print_status()
    cfg.set_units(codeu)
    model.set_solver_config(cfg)
    model.set_particle_mass(pmass)

    # should be number of SPH particles per GPU / 4?
    # seems that it can be quite large...
    model.init_scheduler(int(1e7),1)

    # resize simulation box
    sbmin = (-xmax * 2, -xmax * 2, -xmax * 2)
    sbmax = (xmax * 2, xmax * 2, xmax * 2)
    model.resize_simulation_box(sbmin, sbmax)
    bmin = (-xmax,-xmax,-xmax)
    bmax = (xmax,xmax,xmax)
    # center = (0,0,0)

    # generate model setup
    setup = model.get_setup()

    # gen = setup.make_generator_lattice_hcp_smap(dr, bmin, bmax, [rhoprofile], "spherical", ["r"])
    hcp = setup.make_generator_lattice_hcp(dr, bmin, bmax)

    stretched_hcp = setup.make_modifier_stretch_mapping(
    parent=hcp, rhoprofiles=[rhoprofile], system="spherical", axes=["r"])
    
    setup.apply_setup(stretched_hcp)
    model.set_cfl_cour(C_cour)
    model.set_cfl_force(C_force)

    # convergence for smoothing length
    # model.change_htolerances(coarse=1.3, fine=min(1.3, 1.1))
    # model.timestep()
    # model.change_htolerances(coarse=1.1, fine=min(1.1, 1.1))
    print("hpart: ",ctx.collect_data()["hpart"])
    return ctx, model

def export(model, ctx, dump_prefix, rhotarget):
    folder_name = sham_utilities.get_folder(dump_prefix)
    path_withoutext = f"{folder_name}/{sham_utilities.gen_new_dump_name(dump_prefix)}"
    print(dump_prefix)
    dump_path=f"{path_withoutext}.sham"
    model.dump(dump_path)
    print(f"Dumped {dump_path}")
    # plot_utilities.plot_lattice(ctx,plot_path)
    data = ctx.collect_data()
    mpart = model.get_particle_mass()

    figpos = px_utilities.px_Pos(data)
    # figpos.write_image(f"{path_withoutext}_pos.png")

    figrho = px_utilities.px_rho(data,mpart,rhotarget)

    figrho = px_utilities.px_hist(data)
    # figrho.write_image(f"{path_withoutext}_rho.png")

    print(f"Plotted {path_withoutext}")

# ! Simulation parameters
if __name__ == '__main__':
    dump_prefix = "lattice_stretched_"
    Npart_i = 10
    xmax = 1
    dr = 2*xmax/(Npart_i-1)
    pmass = 1
    rhoprofile = lambda r: 1/((r+0.1)**2)
    # rhoprofile = lambda r: 1
    # rhoprofile = np.sinc


    ctx, model = Model(dr,xmax, pmass, rhoprofile=rhoprofile)
    lastdumpnb, folder_name = handle_dump(dump_prefix)
    export(model, ctx, dump_prefix, rhoprofile)
    print(f"Ended up with {model.get_total_part_count()} particles")

# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./lattice_stretched_shamrock.py 