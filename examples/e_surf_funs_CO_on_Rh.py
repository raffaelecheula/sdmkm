################################################################################
# Raffaele Cheula, LCCP, Politecnico di Milano, cheula.raffaele@gmail.com
################################################################################

from scipy.interpolate import make_interp_spline
from shape.ase_utils import convert_miller_index
from shape.thermochemistry.ab_initio_thermodynamics import (
    surface_energies, 
    surface_energies_funs,
)

################################################################################
# INITIALIZATION
################################################################################

rangemu_CO = [-2.40, -1.20] # [eV]
rangemu_H2 = [-1.20, +0.00] # [eV]

miller_list = ['100', '110', '111', '311', '331']

miller_supp = '111'

e_surf_supp = 0.02459 # [eV/Angstrom**2]

corr_e_surf = True

planes = {} # [eV/Angstrom**2]

################################################################################
# 100 SURFACE
################################################################################

planes['100'] = [
[0.1116, 0.1116, 0.1116, 'Clean 100'],
[0.1131, 0.1036, 0.1131, 'CO 0.0625ML top'],
[0.1149, 0.0959, 0.1149, 'CO 0.125ML top'],
[0.1190, 0.0810, 0.1190, 'CO 0.25ML top'],
[0.1281, 0.0522, 0.1281, 'CO 0.50ML top'],
[0.1538, 0.0400, 0.1538, 'CO 0.75ML Gurney pattern'],
[0.1662, 0.0397, 0.1662, 'CO 0.83ML Jong pattern'],
[0.1149, 0.1149, 0.1054, 'H 0.0625ML bridge'],
[0.1187, 0.1187, 0.0997, 'H 0.125ML bridge'],
[0.1268, 0.1268, 0.0888, 'H 0.25ML bridge'],
[0.1483, 0.1483, 0.0724, 'H 0.50ML bridge'],
[0.1952, 0.1952, 0.0434, 'H 1.00ML bridge'],
[0.3149, 0.3149, 0.0114, 'H 2.00ML bridge'],
[0.1144, 0.1049, 0.1049, 'CO 0.0625ML top H 0.0625ML bridge'],
[0.1187, 0.0998, 0.0998, 'CO 0.125ML top H 0.125ML bridge'],
[0.1350, 0.0971, 0.0971, 'CO 0.25ML top H 0.25ML bridge'],
[0.1586, 0.1207, 0.0827, 'CO 0.25ML top H 0.50ML bridge'],
[0.1753, 0.0994, 0.0994, 'CO 0.50ML bridge H 0.50ML bridge'],
[0.1755, 0.0743, 0.1249, 'CO 0.667ML bridge H 0.333ML bridge']]

################################################################################
# 110 SURFACE
################################################################################

planes['110'] = [
[0.1125, 0.1125, 0.1125, 'Clean 110'],
[0.1135, 0.1068, 0.1135, 'CO 0.0625ML top'],
[0.1147, 0.1012, 0.1147, 'CO 0.125ML top'],
[0.1173, 0.0904, 0.1173, 'CO 0.25ML top'],
[0.1243, 0.0706, 0.1243, 'CO 0.50ML top'],
[0.1565, 0.0492, 0.1565, 'CO 1.00ML shiftedhollow'],
[0.1152, 0.1152, 0.1085, 'H 0.0625ML shortbridge'],
[0.1181, 0.1181, 0.1047, 'H 0.125ML shortbridge'],
[0.1244, 0.1244, 0.0976, 'H 0.25ML shortbridge'],
[0.1387, 0.1387, 0.0851, 'H 0.50ML shortbridge'],
[0.1718, 0.1718, 0.0645, 'H 1.00ML shortbridge'],
[0.2683, 0.2683, 0.0536, 'H 2.00ML shortbridge'],
[0.1158, 0.1091, 0.1091, 'CO 0.0625ML top H 0.0625ML shortbridge'],
[0.1200, 0.1065, 0.1065, 'CO 0.125ML top H 0.125ML shortbridge'],
[0.1287, 0.1018, 0.1018, 'CO 0.25ML top H 0.25ML shortbridge'],
[0.1447, 0.1178, 0.0910, 'CO 0.25ML top H 0.50ML shortbridge'],
[0.1629, 0.1361, 0.0824, 'CO 0.25ML top H 0.75ML shortbridge'],
[0.1567, 0.1030, 0.1030, 'CO 0.50ML shortbridge H 0.50ML shortbridge'],
[0.2052, 0.1516, 0.0979, 'CO 0.50ML top H 1.00ML shiftedhollow']]

################################################################################
# 111 SURFACE
################################################################################

planes['111'] = [
[0.0973, 0.0973, 0.0973, 'Clean 111'],
[0.1007, 0.0897, 0.1007, 'CO 0.0625ML top'],
[0.1043, 0.0824, 0.1043, 'CO 0.125ML top'],
[0.1120, 0.0682, 0.1120, 'CO 0.25ML top'],
[0.1199, 0.0614, 0.1199, 'CO 0.33ML top'],
[0.1448, 0.0572, 0.1448, 'CO 0.50ML top'],
[0.1880, 0.0565, 0.1880, 'CO 0.75ML top+fcc+hcp'],
[0.1017, 0.1017, 0.0907, 'H 0.0625ML fcc'],
[0.1069, 0.1069, 0.0850, 'H 0.125ML fcc'],
[0.1183, 0.1183, 0.0745, 'H 0.25ML fcc'],
[0.1445, 0.1445, 0.0569, 'H 0.50ML fcc'],
[0.1777, 0.1777, 0.0463, 'H 0.75ML fcc'],
[0.2100, 0.2100, 0.0347, 'H 1.00ML fcc'],
[0.1045, 0.0936, 0.0936, 'CO 0.0625ML top H 0.0625ML fcc'],
[0.1135, 0.0915, 0.0915, 'CO 0.125ML top H 0.125ML fcc'],
[0.1352, 0.0913, 0.0913, 'CO 0.25ML top H 0.25ML fcc'],
[0.1695, 0.1256, 0.0818, 'CO 0.25ML top H 0.50ML fcc+hcp'],
[0.1767, 0.0891, 0.1329, 'CO 0.50ML top H 0.25ML fcc']]

################################################################################
# 211 SURFACE
################################################################################

planes['211'] = [
[0.1101, 0.1101, 0.1101, 'Clean 211'],
[0.1125, 0.0970, 0.1125, 'CO 0.083ML top'],
[0.1162, 0.0852, 0.1162, 'CO 0.167ML top'],
[0.1316, 0.0697, 0.1316, 'CO 0.333ML top'],
[0.1549, 0.0619, 0.1549, 'CO 0.50ML top'],
[0.1810, 0.0570, 0.1810, 'CO 0.667ML top'],
[0.2684, 0.0825, 0.2684, 'CO 1.00ML top'],
[0.1167, 0.1167, 0.1012, 'H 0.083ML bridge'],
[0.1239, 0.1239, 0.0929, 'H 0.167ML bridge'],
[0.1419, 0.1419, 0.0799, 'H 0.333ML bridge'],
[0.1597, 0.1597, 0.0667, 'H 0.50ML bridge'],
[0.1800, 0.1800, 0.0561, 'H 0.667ML bridge'],
[0.2269, 0.2269, 0.0410, 'H 1.00ML bridge'],
[0.1192, 0.1037, 0.1037, 'CO 0.083ML top H 0.083ML bridge'],
[0.1311, 0.1001, 0.1001, 'CO 0.167ML top H 0.167ML bridge'],
[0.1550, 0.0930, 0.1240, 'CO 0.333ML top H 0.167ML bridge'],
[0.1525, 0.1216, 0.0906, 'CO 0.167ML top H 0.333ML bridge'],
[0.1797, 0.1177, 0.1177, 'CO 0.333ML top H 0.333ML bridge'],
[0.2302, 0.1682, 0.1063, 'CO 0.333ML top H 0.667ML bridge'],
[0.2224, 0.0985, 0.1604, 'CO 0.667ML top H 0.333ML bridge']]

################################################################################
# 311 SURFACE
################################################################################

planes['311'] = [
[0.1128, 0.1128, 0.1128, 'Clean 311'],
[0.1142, 0.1027, 0.1142, 'CO 0.0625ML top'],
[0.1162, 0.0933, 0.1162, 'CO 0.125ML top'],
[0.1217, 0.0760, 0.1217, 'CO 0.25ML top'],
[0.1562, 0.0646, 0.1562, 'CO 0.50ML top'],
[0.1977, 0.0604, 0.1977, 'CO 0.75ML top'],
[0.2656, 0.0825, 0.2656, 'CO 0.75ML top'],
[0.1175, 0.1175, 0.1060, 'H 0.0625ML bridge'],
[0.1226, 0.1226, 0.0997, 'H 0.125ML bridge'],
[0.1334, 0.1334, 0.0876, 'H 0.25ML bridge'],
[0.1591, 0.1591, 0.0676, 'H 0.50ML bridge'],
[0.1903, 0.1903, 0.0530, 'H 0.75ML bridge'],
[0.2268, 0.2268, 0.0438, 'H 1.00ML bridge'],
[0.1187, 0.1072, 0.1072, 'CO 0.0625ML top H 0.0625ML bridge'],
[0.1266, 0.1037, 0.1037, 'CO 0.125ML top H 0.125ML bridge'],
[0.1446, 0.0988, 0.0988, 'CO 0.25ML top H 0.25ML bridge'],
[0.1821, 0.0906, 0.1363, 'CO 0.25ML top H 0.50ML bridge'],
[0.1816, 0.1358, 0.0901, 'CO 0.50ML top H 0.25ML bridge']]
[0.2172, 0.1257, 0.1257, 'CO 0.50ML top H 0.50ML bridge'],

################################################################################
# 331 SURFACE
################################################################################

planes['331'] = [
[0.1097, 0.1097, 0.1097, 'Clean 331'],
[0.1126, 0.0952, 0.1126, 'CO 0.083ML top'],
[0.1169, 0.0821, 0.1169, 'CO 0.167ML top'],
[0.1344, 0.0647, 0.1344, 'CO 0.333ML top'],
[0.1578, 0.0533, 0.1578, 'CO 0.50ML top'],
[0.1950, 0.0557, 0.1950, 'CO 0.667ML top'],
[0.3800, 0.1711, 0.3800, 'CO 1.00ML top'],
[0.1180, 0.1180, 0.1006, 'H 0.083ML bridge'],
[0.1270, 0.1270, 0.0922, 'H 0.167ML bridge'],
[0.1462, 0.1462, 0.0766, 'H 0.333ML bridge'],
[0.1698, 0.1698, 0.0654, 'H 0.50ML bridge'],
[0.1935, 0.1935, 0.0542, 'H 0.667ML bridge'],
[0.2522, 0.2522, 0.0432, 'H 1.00ML bridge'],
[0.1211, 0.1037, 0.1037, 'CO 0.083ML top H 0.083ML bridge'],
[0.1366, 0.1018, 0.1018, 'CO 0.167ML top H 0.167ML bridge'],
[0.1738, 0.1042, 0.1390, 'CO 0.333ML top H 0.167ML bridge'],
[0.1774, 0.1426, 0.1078, 'CO 0.167ML top H 0.333ML bridge'],
[0.1871, 0.1175, 0.1175, 'CO 0.333ML top H 0.333ML bridge'],
[0.2368, 0.1324, 0.1324, 'CO 0.50ML top H 0.50ML bridge'],
[0.2693, 0.1300, 0.1997, 'CO 0.50ML top H 0.333ML bridge']]

################################################################################
# FUNCTIONS
################################################################################

e_surf_funs = surface_energies_funs(planes      = planes     ,
                                    miller_list = miller_list,
                                    rangemu_A   = rangemu_CO ,
                                    rangemu_B   = rangemu_H2 )

for i in range(len(miller_list)):
    miller_list[i] = convert_miller_index(miller_list[i])

miller_supp = convert_miller_index(miller_supp)

deltamu_CO_vect = [-2.4+i*0.1 for i in range(4)]+[-2.0+i*0.25 for i in range(4)]

e_surf_vect = {}

for facet in miller_list:
    e_surf_vect[facet] = []

for deltamu_CO in deltamu_CO_vect:

    e_surf_list = surface_energies(e_surf_funs = e_surf_funs,
                                   deltamu_A   = deltamu_CO ,
                                   deltamu_B   = -1.2       )
    
    if corr_e_surf is True:
    
        e_surf_list[2] += 0.0016
        e_surf_list[3] += 0.0014
        e_surf_list[4] += 0.0012
        
        if deltamu_CO > -2.2:
            e_surf_list[0] += 0.002*(deltamu_CO+2.2)
            e_surf_list[1] += 0.003*(deltamu_CO+2.2)
            if deltamu_CO > -1.9:
                e_surf_list[3] -= 0.008*(deltamu_CO+1.9)
                e_surf_list[4] -= 0.002*(deltamu_CO+1.9)
            if deltamu_CO > -1.7:
                e_surf_list[3] -= 0.019*(deltamu_CO+1.7)

    for i in range(len(miller_list)):
        facet = miller_list[i]
        e_surf_vect[facet] += [e_surf_list[i]]

e_surf_fun_dict = {}

for facet in miller_list:

    e_surf_fun_dict[facet] = make_interp_spline(deltamu_CO_vect,
                                                e_surf_vect[facet], k = 3)

################################################################################
# END
################################################################################
