
import pandas as pd
from featurebox.featurizers.compositionfeaturizer import DepartElementFeaturizer
from pandas.io import json
import os
from pymatgen.util.string import formula_double_format
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.io.cif import CifFile, CifParser
from pymatgen.io.vasp import Poscar

from tqdm import tqdm

os.chdir("/home/iap13/wcx/bandgapdata/bgdata")
list_name = os.listdir()

from featurebox.data.impot_element_table import element_table

name_and_abbr = element_table.iloc[[0, 1], :]
element_table = element_table.iloc[2:, :]
elemen = element_table[['electronegativity(martynov&batsanov)','electron number']]



dict_all=[]
for i in tqdm(list_name):
    try:
        a = json.read_json(i, orient='index', typ='series')
        cif_str = a["Structure_rlx"]
        del a["Structure_rlx"]

        POSCAR = Poscar.from_string(cif_str)
        ele_den = POSCAR.structure.composition.total_electrons / POSCAR.structure.volume
        composition_mp = POSCAR.structure.composition

        ncom = POSCAR.structure.composition.to_data_dict['unit_cell_composition'].values()

        sym_amt = composition_mp.get_el_amt_dict()
        syms = sorted(sym_amt.keys(), key=lambda sym: get_el_sp(sym).X)
        formula = {s:formula_double_format(sym_amt[s], False) for s in syms}

        departElementProPFeature = DepartElementFeaturizer(elem_data=elemen, n_composition=len(ncom), n_jobs=1,return_type='df' )
        departElement = departElementProPFeature.fit_transform([formula])

        ncom = {"ncom" + "_" + str(n):j  for n,j in enumerate(ncom)}

        for k,v in ncom.items():
            departElement[k] = v
        a = pd.DataFrame(a)
        c = pd.DataFrame(departElement.values.T, index=departElement.columns, columns=departElement.index)
        b = pd.concat((a,c),axis=0)

        b.loc["ele_den"] = ele_den
        b.loc["name"] = i

        dict_all.append(b)
    except:
        pass

data = pd.concat(dict_all, axis=1,)
data = data.T
data.index = data["name"]
data.to_csv("../other_data.csv")




