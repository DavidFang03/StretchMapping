if __name__ == "__main__":

    ###########################
    folder_path = "./outputs/2balls_tillotson_2000k_mm_cd10_uc_skewv2_000"
    ##########################
    from balls import ShamPlot
    import sham_utilities as shu
    import glob
    import shamrock

    shu.handle_dump(folder_path=folder_path, clear=True, onlyext=".png")
    dumps = glob.glob(f"{folder_path}/*.sham")

    if not shamrock.sys.is_initialized():
        shamrock.change_loglevel(1)
        shamrock.sys.init("0:0")

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    for i in range(len(dumps)):
        dump = dumps[i]
        print(f"replaying {i} : {dump}")
        model.load_from_dump(dump)
        if i == 0:
            plot = ShamPlot(model, ctx)
        plot.update_pvplot()
        img_path = dump.replace(".sham", ".png")
        plot.screenshot(img_path)

# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./replay.py
