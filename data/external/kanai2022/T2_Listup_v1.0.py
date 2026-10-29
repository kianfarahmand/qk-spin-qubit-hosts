# -*- coding: utf-8 -*-
"""
version 0 : Created on Thu Nov 12 21:38:34 2020
T2_Listup_v1.0.py
Shun Kanai @Tohoku University with Pf. D.D. Awschalom Group, UChicago
skanai@tohoku.ac.jp
"""
from pymatgen import MPRester, Structure
# from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.elasticity import ElasticTensor
# from pymatgen.analysis.thermochemistry import ThermoData
MY_API_KEY="*******************"
import csv
import numpy as np
Nan = np.nan
# import warnings
# warnings.simplefilter('ignore', category=RuntimeWarning)

#loadisotope
def loadisotope():
    import csv
    with open('isotopetable.csv','r',encoding="utf-8") as f:
        data = csv.reader(f)
        dataall = [row for row in data]
    return dataall

#return T2list of isotopes in an element
def PyT2Ele(dataall, element, density):
    A = 1.50e18 # sec cm-3
    S = 1
    f_g = -1.65 #exponent for gN
    f_I = -1.09 #exponent for IN
    T2list = []
    MaxDens = 0.0
    MaxDensEle = []

    if (element == 'Ce' or element == 'Ar'): #stable but no isotopes with s > 0
        T2 = float('inf')
        T2list = [T2]
        MaxDensEle = element
    else :
        for row in dataall:
            if (row[0] == element):
                abund = float(row[6]) * 0.01
                T2list.append(A * S * (abs(float(row[5])))**f_g * (float(row[4]))**f_I / abund / density)
                temp2175555 = abund * density
                if (MaxDens < temp2175555):
                    MaxDens = temp2175555
                    MaxDensEle = str(row[3]) +  element
        if (T2list == []):
            T2list = [1e-20]
        n = 2
        T2 = (sum(np.power(T2list,-n))) ** (-1/n)
    return T2, T2list, MaxDens, MaxDensEle  #T2 in sec

"""
main
"""
#load isotope table
dataall = loadisotope()

#importdata
savename = 'data/all(1eV).csv'
# criteria_properties = {"band_gap":{"$gt": 0.01},"elements": {"$all": ["O"]}, "e_above_hull": {"$lte": 0}, "nelements": {"$lte": 3}}
criteria_properties = {"band_gap":{"$gt": 1}, "e_above_hull": {"$lte": 0}, "nelements": {"$lte": 10}}
# print(criteria_properties)
output = ['pretty_formula', "band_gap","unit_cell_formula", "volume", 'nelements', 'task_id', 'spacegroup', 'cif','elasticity','elastic_tensor', 'total_magnetization']
with MPRester( MY_API_KEY ) as m:
    mp_entries = m.query(criteria=criteria_properties, properties=output)
print(len(mp_entries),"materials found!")
mat_all = len(mp_entries)

#formatting & T2 calc
datawrite = []
for mat in mp_entries:
    # print(mat)
    elementss = mat['unit_cell_formula'];

    if mat['elasticity'] is None:
        DebyeTemp= Nan
        K_Reuss = Nan
    else:
        K_Reuss = mat['elasticity']['K_Reuss']
        structurea = Structure.from_str(mat['cif'], 'cif')
        elastic_tensor = ElasticTensor.from_voigt(mat['elasticity']['elastic_tensor'])
        try:
            DebyeTemp = elastic_tensor.debye_temperature(structurea)
        except:
            DebyeTemp = Nan
    # print(tempa)

    materialsa = list(elementss.items())
    # print(mat['pretty_formula'],"Eg={:.2f}eV".format(mat['band_gap']),mat['task_id'])
    T2inComp = []
    flag_chalco = 0 #0: others, 1: sulfide, 2: oxides, 3: suflates, 4: other calcogenides
    flag_U = 0
    MaxDensEle = []
    MaxDens = 0.0

    for i in range(mat['nelements']):
        elementa = materialsa[i][0]
        density = materialsa[i][1]/(mat['volume']*1e-30)*1e-6
        temp2734507 = PyT2Ele(dataall, materialsa[i][0], density)
        T2list = temp2734507[1]
#       print(temp2734507[2])
        if (MaxDens < temp2734507[2]):
            MaxDensEle = temp2734507[3]
            MaxDens = temp2734507[2]
        # print(elementa, "\tn={:.2e}cm^-3".format(density), "\tT2={:.2e}sec".format(T2))
        T2inComp.extend(T2list)
        flag_chalco += (elementa == 'O')*8 + (elementa == 'S')*4 + (elementa == 'Se') *2 + (elementa == 'Te') *1
        flag_U += (elementa == 'U')
    n = 2
    # print(T2inComp)
    T2 = (sum(np.power(T2inComp,-n))) ** (-1/n)

    # print(mat['pretty_formula'],"Eg={:.2f}eV".format(mat['band_gap']),mat['task_id'],"T2={:.2e}sec".format(T2))
    datarow = [mat['pretty_formula'],mat['task_id'],mat['band_gap'],mat['total_magnetization'],mat['nelements'],T2,flag_chalco,flag_U,DebyeTemp,MaxDens,MaxDensEle,mat['spacegroup']['crystal_system'],mat['spacegroup']['point_group'],K_Reuss]
    datawrite.append(datarow)

#sort by T2
datawrite = sorted(datawrite, reverse=True, key=lambda x: x[5])

#save data
with open(savename, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['material','id','band gap (eV)', 'total_magnetization','nelements','T2 (s)','chalcogenides?(8: oxides, 4: sulfides, 2: selenides, 1: tellurates, 0: others)','uranium?','Debye temp (K)', 'MaxDensity (cm-3)', 'MaxDensityNuclei','space group','point group','bulk modulous (GPa)'])
with open(savename, 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(datawrite)
