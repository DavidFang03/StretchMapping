def get_folder_prefix(dump_prefix):
    folder_prefix = dump_prefix[:-1]
    return f"outputs/{folder_prefix}"


def get_new_folder(dump_prefix):
    import glob

    folder_prefix = get_folder_prefix(dump_prefix)
    new_folder_name = f"{folder_prefix}_000"
    iii = 0
    while len(glob.glob(new_folder_name)) > 0:
        iii += 1
        new_folder_name = f"{folder_prefix}_{iii:0>3}"
    return new_folder_name


def get_last_folder(dump_prefix):
    import glob

    folder_prefix = get_folder_prefix(dump_prefix)
    new_folder_name = f"{folder_prefix}_000"
    iii = 0
    while len(glob.glob(f"{folder_prefix}_{iii+1:0>3}")) > 0:
        iii += 1
        new_folder_name = f"{folder_prefix}_{iii:0>3}"
    return new_folder_name


def get_dump_name(dump_prefix, idump):
    """
    generates dump file path
    """
    return dump_prefix + f"{idump:04}"


def get_last_dump(dump_prefix):
    """
    get number of last dump ?
    """
    import glob

    folder = get_last_folder(dump_prefix)
    pattern = f"{folder}/{dump_prefix}*.sham"
    res = glob.glob(pattern)
    # print("globbing: ",pattern,res)
    num_max = -1
    for f in res:
        try:
            dump_num = int(f[len(f"{folder}/{dump_prefix}") : -5])
            if dump_num > num_max:
                f_max = f
                num_max = dump_num
        except:
            pass
    if num_max == -1:
        return None
    else:
        return num_max


def get_last_plot(dump_prefix):
    """
    get number of last plot ?
    """
    import glob

    res = glob.glob(dump_prefix + "*.png")
    num_max = -1
    for f in res:
        try:
            plot_num = int(f[len(dump_prefix) : -4])
            if plot_num > num_max:
                num_max = plot_num
        except:
            pass
    return num_max


def get_last_dump_path(dump_prefix):
    folder = get_last_folder(dump_prefix)
    lastdumpnb = get_last_dump(dump_prefix)
    if lastdumpnb is None:
        return None
    dumpname = get_dump_name(dump_prefix, lastdumpnb)
    return f"{folder}/{dumpname}.sham"


def gen_new_dump_name(dump_prefix):
    lastdumpnb = get_last_dump(dump_prefix)
    if lastdumpnb is None:
        lastdumpnb = 0
    else:
        lastdumpnb += 1
    return get_dump_name(dump_prefix, lastdumpnb)


def gen_new_path_withoutext(dump_prefix):
    folder_name = get_last_folder(dump_prefix)
    path_withoutext = f"{folder_name}/{gen_new_dump_name(dump_prefix)}"
    return path_withoutext
