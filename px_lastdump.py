import shamrock
import numpy as np
import os
import sham_utilities
import px_utilities


shamrock.enable_experimental_features()


def load_from_json(json_path):
    import json

    with open(json_path, "r") as fp:
        sample = json.load(fp)
    return sample


def plot(model, ctx, rhotarget, inputparams, img_path):
    data = ctx.collect_data()
    mpart = model.get_particle_mass()
    t = model.get_time()
    fig = px_utilities.px_3d_and_rho(data, rhotarget, mpart, t, inputparams, img_path)

    print("I will write this image in", img_path)
    fig.write_image(img_path)

    return fig


# ! Simulation parameters
if __name__ == "__main__":
    rhoprofile = lambda r: 1 / np.sqrt(1 + (r**2) / 3)
    # rhoprofile = lambda r: 1/np.sqrt(1+r**2)
    # rhoprofile = lambda r: 1/((r+0.1)**2)
    # rhoprofile = lambda r: 1/((r+0.01)**2)
    # rhoprofile = lambda r:
    # rhoprofile = lambda r: np.sinc(r)
    # rhoprofile = np.sinc
    if not shamrock.sys.is_initialized():
        shamrock.change_loglevel(1)
        shamrock.sys.init("0:0")

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    folder_name = "outputs/test_n1_1e2_SG_002"
    inputparams = load_from_json(f"{folder_name}/inputparams.json")

    if inputparams["target"] == "sinc":
        rhoprofile = lambda r: np.sinc(r / inputparams["xmax"])
    elif inputparams["target"] == "1/np.sqrt(1+(r**2)/3)":
        rhoprofile = lambda r: 1 / np.sqrt(1 + (r**2) / 3)

    import glob
    from pathlib import Path

    all_dumps = glob.glob(f"{folder_name}/*.sham")
    for dump_path in all_dumps:
        dump_prefix = f"{Path(dump_path).stem}_"
        model.load_from_dump(dump_path)
        img_path = dump_path.replace("sham", "png")

        plot(
            model,
            ctx,
            rhoprofile,
            inputparams,
            img_path,
        )

    print(f"Ended up with {model.get_total_part_count()} particles")
    pattern_png = f"{folder_name}/*.png"
    filemp4 = img_path.replace("png", "mp4")
    px_utilities.movie(pattern_png, filemp4, fps=px_utilities.compute_fps(inputparams))


# ./shamrock --sycl-cfg 0:0 --loglevel 10 --rscript ./lattice_stretched_shamrock.py
