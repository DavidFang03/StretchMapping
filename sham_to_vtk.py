import shamrock
from test_fermi import initModel


import glob
import os
import json

folder = "tillotson_200k_mm_cd10_rescaled_000"
shamfolder = f"./outputs/{folder}"
vtkfolder = f"./vtk/{folder}"
files_list = []

try:
    os.mkdir(vtkfolder)
except OSError as error:
    user_agree = input(
        f"Will remove the entire {vtkfolder} folder ({len(glob.glob(f"{vtkfolder}/*.vtk"))} dumps) (y/n)"
    )
    if user_agree == "y":
        for f in glob.glob(f"{vtkfolder}/*"):
            os.remove(f)
    else:
        exit()

for shamdump in glob.glob(f"{shamfolder}/*.sham"):
    model, ctx, _ = initModel()
    print(shamdump)
    model.load_from_dump(shamdump)
    t = model.get_time()
    dumpname = os.path.basename(shamdump)
    vtkname = dumpname.replace(".sham", ".vtk")
    vtk_path = f"{vtkfolder}/{vtkname}"
    model.do_vtk_dump(vtk_path, True)

    files_list.append({"name": vtkname, "time": t})


series_path = f"{vtkfolder}/{folder}.vtk.series"
output_data = {"file-series-version": "1.0", "files": files_list}
with open(series_path, "w") as f:
    json.dump(output_data, f, indent=2)

print("done")

# ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript ./sham_to_vtk.py
