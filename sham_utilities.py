from rich.console import Console

console = Console()


def get_folder_path(dump_prefix):
    folder_path = dump_prefix[:-1]
    return f"./outputs/{folder_path}"


def get_new_folder(dump_prefix):
    import glob

    folder_path = get_folder_path(dump_prefix)
    new_folder_name = f"{folder_path}_000"
    iii = 0
    while len(glob.glob(new_folder_name)) > 0:
        iii += 1
        new_folder_name = f"{folder_path}_{iii:0>3}"
    return new_folder_name


def get_last_folder(dump_prefix):
    import glob

    folder_path = get_folder_path(dump_prefix)
    new_folder_name = f"{folder_path}_000"
    iii = 0
    while len(glob.glob(f"{folder_path}_{iii+1:0>3}")) > 0:
        iii += 1
        new_folder_name = f"{folder_path}_{iii:0>3}"
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


def handle_dump(script_file, *, dump_prefix=None, folder_path=None, clear=True, onlyext=""):
    """
    Only kwargs.
    If dump_prefix is given : remove the content of the last folder or creates new folder
    If folder_path is given : remove the content of this specific folder or creates it

    :param dump_prefix: Description
    :param folder_name: Description
    :param overwrite: Description
    :param onlyext: ".png" for example
    """
    import os
    import glob

    def clear_folder(folder_path, onlyext):
        if not os.path.isdir(folder_path):
            raise Exception(f"{folder_path} doesn't exist, cannot clear it")
        files_to_remove = glob.glob(f"{folder_path}/*{onlyext}")

        warning_text = "[r]ENTIRE[/r] content" if onlyext == "" else f"{onlyext} files"

        user_agree = console.input(
            f"Will remove the {warning_text} of {folder_path} ({len(files_to_remove)} files) (y/n)"
        )
        if user_agree == "y":
            for f in files_to_remove:
                os.remove(f)

    def create_folder(folder_path):
        try:
            os.mkdir(folder_path)
        except OSError:
            console.print(
                f"[r]WARNING[/r] directory exists already, probably not clean: {folder_path} "
            )

    def keep_copy(folder_path, dump_prefix, script_path):
        copy_name = f"{folder_path}/{dump_prefix}.py"
        i = 0
        while os.path.isfile(copy_name):
            i += 1
            copy_name = f"{folder_path}/{dump_prefix}_{i}.py"

        command = f"cp {os.path.abspath(script_path)} {copy_name}"
        console.print("I'm keeping a copy of your runscript", command)
        os.system(command)

    try:
        os.mkdir("outputs")
    except OSError:
        console.print("ok outputs exist")

    if folder_path is not None:
        # if folder_path is given
        if clear and os.path.isdir(folder_path):
            clear_folder(folder_path, onlyext)
        else:
            console.print("Creating a [bold]new[/bold] folder")
            create_folder(folder_path)
    else:
        # if prefix is given
        last_eventual_folder = get_last_folder(dump_prefix)
        if clear and os.path.isdir(last_eventual_folder):
            clear_folder(last_eventual_folder, onlyext)
        else:
            create_folder(last_eventual_folder)
        folder_path = last_eventual_folder

    keep_copy(folder_path, dump_prefix, script_file)
    console.print(f"{folder_path} is ready to handle your new instructions")
    return folder_path
