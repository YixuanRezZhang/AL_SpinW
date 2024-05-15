import numpy as np
import matplotlib.pyplot as plt
from pyspinw import Matlab
import argparse
import pickle


m = Matlab('/shared/apps/matlab/R2022b')

parser = argparse.ArgumentParser(description='Input J list')
parser.add_argument('--save_name', type=str, help='name of the file')
parser.add_argument('--J_list', type=float, nargs='+', help='List of J values separated by spaces')
args = parser.parse_args()
name = args.save_name
LJ = args.J_list


def cal_sw(LJ, name):

    J, Jp, Jpp, Jc = LJ
    lacuo = m.sw_model('squareAF', np.array([J-Jc/2, Jp-Jc/4, Jpp])/2, 0)
    lacuo.unit_cell['S'] = np.array([[1/2]])
    
    Zc = 1.18
    
    Qlist = [[3/4, 1/4, 0], [1/2, 1/2, 0], [1/2, 0, 0], [3/4, 1/4, 0], [1, 0, 0], [1/2, 0, 0], 100]
    Qlab  = ['P', 'M', 'X', 'P', '\Gamma', 'X']
    
    lacuoSpec = lacuo.spinwave(Qlist, 'hermit', False)
    lacuoSpec['omega'] = lacuoSpec['omega']*Zc
    lacuoSpec = m.sw_neutron(lacuoSpec)
    lacuoSpec = m.sw_egrid(lacuoSpec,'component','Sperp')
    lacuoSpec = m.sw_instrument(lacuoSpec, dE=35)

    with open(f'{name}.pickle', 'wb') as f:
        pickle.dump(lacuoSpec['swConv'], f)

cal_sw(LJ, name)