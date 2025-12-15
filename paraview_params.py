import json
from paraview import vtk

def format_param(key, value):
    if key in ["y0", "n"]:
        valuestr = value
    elif key in ["mtot_target"]:
        valuestr = f"{value:.3f}"
    elif isinstance(value, float):
        if value > 10 or value < 0.1:
            valuestr = f"{value:.1e}"
        else:
            valuestr = f"{value:.1f}"
    else:
        valuestr = value
    return valuestr


def format_inputparams(input_params):
    string = ""
    for key, value in input_params.items():
        if isinstance(value, dict):
            line = ""
            line += f"{key}[name]={format_param("name",value["name"])} \n"
            for param, param_value in value["values"].items():
                line += f"{key}[{param}]={format_param(param,param_value)} \n"
        else:
            valuestr = format_param(key, value)
            line = f"{key}: {valuestr} \n"
        string += line
    return string

try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except IOError:
    data = {"Erreur": "Fichier introuvable"}

txt = format_inputparams(data)

col = vtk.vtkStringArray()
col.SetName("InfoText") # Nom de la colonne
col.InsertNextValue(txt)

output.GetRowData().AddArray(col)