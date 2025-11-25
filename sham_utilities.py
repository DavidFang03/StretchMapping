def get_folder(dump_prefix):
    folder_name = dump_prefix[:-1]
    return folder_name

def get_dump_name(dump_prefix,idump):
    '''
    generates dump file path
    '''
    return dump_prefix + f"{idump:04}"

def get_last_dump(dump_prefix):
    '''
    get number of last dump ?
    '''
    import glob
    folder = get_folder(dump_prefix)
    pattern = f"{folder}/{dump_prefix}*.sham"
    res = glob.glob(pattern)
    print(pattern,res)
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
    '''
    get number of last plot ?
    '''
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
    folder = get_folder(dump_prefix)
    lastdumpnb = get_last_dump(dump_prefix)
    dumpname = get_dump_name(dump_prefix, lastdumpnb)
    return f"{folder}/{dumpname}.sham"

def gen_new_dump_name(dump_prefix):
    lastdumpnb = get_last_dump(dump_prefix)
    if lastdumpnb is None:
        lastdumpnb = 0
    else :
        lastdumpnb +=1
    return get_dump_name(dump_prefix, lastdumpnb)