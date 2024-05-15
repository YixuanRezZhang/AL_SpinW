#!/usr/bin/env python
# coding: utf-8

import os, shutil, time, pickle, sys

import numpy as np
import traceback

normal_turbo=True
if normal_turbo:
    from software.turbo import TurboM
else:
    from software.turbo_bwo import TurboM
    
from software.BWO.Python import BWO
from software.MantidData import mantidOutput
from pyspinw import Matlab

import torch
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, RBFKernelGrad, SpectralDeltaKernel, SpectralMixtureKernel

from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP, HeteroskedasticSingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.test_functions.base import BaseTestProblem

device, dtype = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'), torch.double
print(device)
root_path = os.getcwd()+'/'

### Reading Exper data and Exp_setup for simulation
def generate_exp_k_path(path_descriptions):
    'path description: list(K_path: str([[h1,k1,l1], [h2,k2,l2]]), Value: np.ndarray([h_value, k_value ,l_vaule]), grid: int(number))'
    'example usage: ["[[-h-1, 0, 0.5], [0, k+1, 0.5]]", [1, 1, 0], 300]] or ["[[1, -k, 0.5], [1, k, 0.5]]", [0, 2, 0], 300]'

    desc = path_descriptions[0]
    Kv = path_descriptions[1]
    path = eval(desc, {"h": Kv[0], "k": Kv[1], "l": Kv[2]})
    path.append(path_descriptions[2])

    return path

paths = ["[[1, -k, 0.5], [1, k, 0.5]]", "[[-h, 2, 0.5], [h, 2, 0.5]]", "[[-h-1, 0, 0.5], [0, k+1, 0.5]]"]

### In future the "txt file" can be subsitute to the experiment measurement function or the interface to the equipment
Expr = {0:{'Exp_info':['1K0E_slice_rotation_Ei70meV_5K.txt', paths[0]]}, 1:{'Exp_info':['H20E_slice_rotation_Ei70meV_5K.txt', paths[1]]}, 2:{'Exp_info':['Hm1Hp10E_slice_rotation_Ei70meV_5K.txt', paths[2]]}}

### part for furture application
Expr_grid = [None, None] #[Energy_grid, Kpath_grid]
    
for i in Expr.keys():
    testfile = f"{root_path}files/data/{Expr[i]['Exp_info'][0]}"
    MO = mantidOutput(testfile)
    K = max(np.abs(np.min(np.round(MO.XX, 4))), np.abs(np.max(np.round(MO.XX, 4))))
    Expr[i]['Exp_info'].append(K)
    Expr[i]['Exp_Kgrid'] = [generate_exp_k_path([Expr[i]['Exp_info'][1], [K,K,K], len(MO.XX) if Expr_grid[0] is None else Expr_grid[0]]), K]
    Expr[i]['Exp_Egrid'] = [np.min(np.round(MO.YY, 4)), np.max(np.round(MO.YY, 4)), len(MO.YY) if Expr_grid[1] is None else Expr_grid[1]]
    
### example output: {1: {'Exp_info': ['1K0E_slice_rotation_Ei70meV_5K.txt', '[[1, -k, 0.5], [1, k, 0.5]]', 2.95], 'Exp_Kgrid': [[[1, -2.95, 0.5], [1, 2.95, 0.5], 60], 2.95], 'Exp_Egrid': [-4.625, 59.875, 87]}, 2: {'Exp_info': ['H20E_slice_rotation_Ei70meV_5K.txt', '[[-h, 2, 0.5], [h, 2, 0.5]]', 3.15], 'Exp_Kgrid': [[[-3.15, 2, 0.5], [3.15, 2, 0.5], 64], 3.15], 'Exp_Egrid': [-4.625, 59.875, 87]}, 3: {'Exp_info': ['Hm1Hp10E_slice_rotation_Ei70meV_5K.txt', '[[-h-1, 0, 0.5], [0, k+1, 0.5]]', 0.975], 'Exp_Kgrid': [[[-1.975, 0, 0.5], [0, 1.975, 0.5], 40], 0.975], 'Exp_Egrid': [-4.625, 59.875, 87]}}

### Simulation_setup
img_shape = [500, 500] #[Energy_grid, Kpath_grid]
exp_img_shape = [[Expr[i]['Exp_Egrid'][2], Expr[i]['Exp_Kgrid'][0][2]] for i in Expr.keys()]
print(exp_img_shape)

# simulation Q and E for SpinW Simulation
Q_paths = [[Expr[i]['Exp_Kgrid'][0][0], Expr[i]['Exp_Kgrid'][0][1], img_shape[1]] for i in Expr.keys()] 
E_grid = [[Expr[i]['Exp_Egrid'][0], Expr[i]['Exp_Egrid'][1], img_shape[0]] for i in Expr.keys()]
energy_range = [[Expr[i]['Exp_Egrid'][0], Expr[i]['Exp_Egrid'][1]] for i in Expr.keys()]

# Experiment Q and E
Exp_Q_paths = [Expr[i]['Exp_Kgrid'][0] for i in Expr.keys()]
Exp_E_paths = [Expr[i]['Exp_Egrid'] for i in Expr.keys()]


### Algorithm_setup
max_evals = 500
m = Matlab(matlab_path="/shared/apps/matlab/R2023a", matlab_version='R2023a')
csb = m.spinw()
csb.genlattice('lat_const',[3.5126, 4.7457, 7.9171],'angled',[90, 90, 90],'sym','P m m n')
csb.addatom('label','Cr3+','r',[1/4, 3/4, 0.37193],'S',3/2,'color','gray')
csb.gencoupling('maxDistance',10)

#--------------------------------------------------------- Start of the Main functions --------------------------------------------------------#
### Read_Write_Function
class IO(object):
    
    def __init__(self, root_path, expr_num, experiment=Expr, img_shape=img_shape, output_file_name='out.pickle', kf_file='stat_model.pickle', sample_file='SW_point.txt'):

        # simulation Q and E for SpinW Simulation

        Expr = experiment[expr_num]
        Q_paths = [Expr['Exp_Kgrid'][0][0], Expr['Exp_Kgrid'][0][1], img_shape[1]] 
        E_grid = [Expr['Exp_Egrid'][0], Expr['Exp_Egrid'][1], img_shape[0]]
        
        # Experiment Q and E
        Exp_Q_paths = Expr['Exp_Kgrid'][0]
        Exp_E_paths = Expr['Exp_Egrid']
        
        self.root_path = root_path
        self.expr_num = expr_num
        self.experiment = Expr
        self.K = Expr['Exp_Kgrid'][1]
        self.simu_Q = Q_paths
        self.simu_E = E_grid
        self.expr_Q = Exp_Q_paths
        self.expr_E = Exp_E_paths
        self.output_file_name = output_file_name
        self.kf_file = kf_file
        self.sample_file = sample_file

    def read_output(self, folder):

        with open(f'{self.root_path}{folder}/{self.output_file_name}', 'rb') as file:
            para, sw = pickle.load(file)

        return para, sw

    def read_output_KF(self, folder):

        with open(f'{self.root_path}{folder}/{self.kf_file}', 'rb') as file:
            kf_sw = pickle.load(file)

        return kf_sw

    def read_experimet(self):
        
        expfile = f"{root_path}files/data/{self.experiment['Exp_info'][0]}"
        MO = mantidOutput(expfile)

        return MO.II
    
    def read_SW(self, kf_name=None):

        if kf_name is None:
            kf_name = f'{self.expr_num}_{self.sample_file}'
            
        with open(f'{self.root_path}{kf_name}', 'r') as f:
            SWl = f.readlines()

        X = []
        Y = []
        for i in SWl:
            info = i.split()
            _x = [float(info[0]), float(info[1])]
            X.append(_x)
            Y.append(float(info[2]))

        norm_X = np.array(X)
        norm_Y = np.array(Y)

        return [norm_X, norm_Y]

    ### Seems like current in the code only have E/Q to index transfermation. the reverse may be needed in sampling part
    
    def data_index_transformation(self, data, direction='expdata_2_simuindex'):
        '''Currently only serves for index transformation to (0,500) and (0,300)'''
        '''The index range is changing with the Quanty input settings'''
        '''input 2*N --> output 2*N'''
        '''data type numpy N*2 matrix [(x0_1,x0_2), (x1_1,x1_2), ...]''' 
        '''data format: [Energy, K]'''
        
        if type(data)!=np.ndarray:
            data = np.array(data)
            
        if len(data.shape) < 2:
            data = np.expand_dims(data, axis=0)
            
        new_array = np.zeros(data.shape)

        ### for Hamiltonian fitting, to get simu_index from experiment [Energy, Q]
        if direction == 'expdata_2_simuindex':     
            assert (self.simu_E[0] <= data[:,0]).all() and (data[:,0] <= self.simu_E[1]).all()
            assert (-self.K <= data[:,1]).all() and (data[:,1] <= self.K).all()
            
            # 1. Exp energy to simu index and Exp Kgrid to simu index
            energy_to_simu_index = (data[:,0]-self.simu_E[0])/(self.simu_E[1]-self.simu_E[0])*self.simu_E[2]
            Q_to_simu_index = (data[:,1]-(-self.K))/(self.K-(-self.K))*self.simu_Q[2]
            
            new_array[:,0] = np.array([np.around(i, 0) for i in energy_to_simu_index])
            new_array[:,1] = np.array([np.around(i, 0) for i in Q_to_simu_index])
            
        ### for sampling [energy, K], to get experiment [Energy, Q] from simu_index
        elif direction == 'simuindex_2_expdata':
            assert (data[:,0] <= self.simu_E[2]).all() and (data[:,1] <= self.simu_Q[2]).all()
            
            # 2. simu index to Exp energy and simu index to Exp Kgrid
            simu_index_to_energy = (data[:,0]/self.simu_E[2])*(self.simu_E[1]-self.simu_E[0])+self.simu_E[0]
            simu_index_to_Q = (data[:,1]/self.simu_Q[2])*(self.K-(-self.K))-self.K

            new_array[:,0] = simu_index_to_energy
            new_array[:,1] = simu_index_to_Q

        ### for mimic the experiment measurement, may be deprecated in furture version when access to real experimental test
        elif direction == 'expdata_2_expindex':
            assert (self.expr_E[0] <= data[:,0]).all() and (data[:,0] <= self.expr_E[1]).all()
            assert (-self.K <= data[:,1]).all() and (data[:,1] <= self.K).all()

            # 3. Exp energy to Exp index and Exp Kgrid to Exp index
            energy_to_expr_index = (data[:,0]-self.expr_E[0])/(self.expr_E[1]-self.expr_E[0])*self.expr_E[2]
            Q_to_expr_index = (data[:,1]-(-self.K))/(self.K-(-self.K))*self.expr_Q[2]
            
            new_array[:,0] = np.array([np.around(i, 0) for i in energy_to_expr_index])
            new_array[:,1] = np.array([np.around(i, 0) for i in Q_to_expr_index])

        elif direction == 'expindex_2_expdata':
            assert (data[:,0] <= self.expr_E[2]).all() and (data[:,1] <= self.expr_Q[2]).all()
            
            # 2. Exp index to Exp energy and Exp index to Exp Kgrid
            expr_index_to_energy = (data[:,0]/self.expr_E[2])*(self.expr_E[1]-self.expr_E[0])+self.expr_E[0]
            expr_index_to_Q = (data[:,1]/self.expr_Q[2])*(self.K-(-self.K))-self.K

            new_array[:,0] = expr_index_to_energy
            new_array[:,1] = expr_index_to_Q

        return new_array

    def database_convert(self, i, index):

        X = []   ###   Parameters
        Y = []

        file_path = 'nn_files/'+str(i)+'/previous/' 

        for j in index:
            if os.path.isfile(f'{self.root_path}{file_path}{j}/{self.output_file_name}'):
                para, sw = self.read_output(file_path+str(j))
                X.append(para)
                Y.append(sw)

        print('nn_file reading finish')
        parameters, sw_result = X, Y

        model_res = []
        for QQ in range(len(self.simu_Q)):
            model_q = np.stack([i[QQ] for i in sw_result])
            mean_q = np.mean(model_q, axis=0)
            std_q = np.std(model_q, axis=0)
            model_res.append([mean_q, std_q])

        os.chdir(self.root_path)
        with open('database_nn.pickle', 'wb') as f:
            pickle.dump([parameters, sw_result], f)

        with open(f'{self.kf_file}', 'wb') as f:
            pickle.dump(model_res, f)
            

#--------------------------------------------------------- Experimental measurement ---------------------------------------------------------#
####### mimic the experiment test to pretend we get experiment results(but actually is the simulation results with a fixed parameter) #######
def get_SW(X, Q):
    
    if X.ndim <= 1:
        X = X.unsqueeze(0)

    io = IO(root_path, Q)
    LX_index = io.data_index_transformation(X, 'expdata_2_expindex')
    LY = io.read_experimet()

    os.chdir(root_path)
    
    ###### output Y ######
    Lit = []
    Lit_y = []   
    for num, lx in enumerate(LX_index):
        NX = [round(i.item(), 5) for i in X[num]]
        NY = LY[tuple([int(i) for i in lx])]
        NY = max(NY, 0)

        ### np.nan processing, further updation will be thought during tests
        if not np.isnan(NY) and NX[0] >= 5:
            Lit_y.append(NY)
            Lit.append([NX[0], NX[1], NY])

    with open(f'{Q}_SW_point.txt', 'a') as f:
        for iter_num,j in enumerate(Lit):
            f.write(str(j[0])+'   '+str(j[1])+'   '+str(j[2])+'   '+str(iter_num)+'\n')

    return torch.tensor(Lit_y, dtype=dtype, device=device).unsqueeze(-1)
    

###### read the last round parameter fitted by BO_search and calculate its SW ######
def read_SW_para(root_path=root_path, img_shape=img_shape, energy_range=energy_range, Q_paths=Q_paths, file='para_results.txt', folder_name='parameters/'):
    
    ###### get quanty parameter and data ######
    folder_init = int(os.popen('cat '+root_path+file+' | wc -l').read())

    folder = str(folder_init) 
    if not os.path.isdir(root_path+folder_name):
        os.mkdir(root_path+folder_name)
    if not os.path.isdir(root_path+folder_name+folder):
        os.mkdir(root_path+folder_name+folder)

    with open(root_path+file, 'r') as f:
        para_line = f.readlines()
    x = [float(i) for i in para_line[-1].split()[1].split('_')]
    
    # Redirect standard output and error to null device
    sys.stdout = open(os.devnull, 'w')
    #sys.stderr = open(os.devnull, 'w')

    ### Calculate target function
    res = func(x, img_shape, energy_range, Q_paths, fit=False, csb=csb)

    # Restore standard output and error
    sys.stdout = sys.__stdout__
    #sys.stderr = sys.__stderr__

    with open(root_path+folder_name+folder+'/out.pickle', 'wb') as f:
        pickle.dump([x, res], f)
    os.chdir(root_path)


##  function of physical model part
def func(X, img_shape, energy_range, Q_paths, fit=True, csb=csb):
    
    ##---------------part to change matlab spinw function --> #input x#---------------##

    #J1, J2, J3, J4, J5, J6, J7, J8, Aniso1, Aniso2, DM1, dE = X
    J1, J2, J3, J4, J5, J7, J8, DM1, dE = X
    #J1, J2, J3, J4, J5, J7, J8, dE = X
    
    Jfile = {'J1':J1, 'J2':J2, 'J3':J3, 'J4':J4, 'J5':J5, 'J6':0, 'J7':J7, 'J8':J8}   # 8 parameters
    for count, i in enumerate(Jfile.keys()):
        csb.addmatrix('value',Jfile[i],'label',i)
        csb.addcoupling('mat',i,'bond',count+1)
    
    csb.addmatrix('value', m.diag([0.0013, -0.0076, 0.0]),'label','Aniso')  # 2 parameters x, y
    csb.addaniso('Aniso')
    
    csb.addmatrix('label','DM1','value',1,'color','b')
    csb.addcoupling('mat','DM1','bond',1)
    csb.setmatrix('mat','DM1','pref',(DM1, 0, 0)) # 1 Parameters

    csb.genmagstr('mode','direct','S',[[0, 0], [1, 1], [0, 0]],'n',[0, 0, 1],'k',[0,0,1/2])
    csb.table('mat')

    results = []
    for num, QQ in enumerate(Q_paths):
        if fit:
            try:
                csbSpec = csb.spinwave(QQ,'formfact',True)
                csbSpec = m.sw_neutron(csbSpec)
                #csbSpec = m.sw_egrid(csbSpec,'component','Sperp', 'imagChk', False, 'hermit', False)
                csbSpec = m.sw_egrid(csbSpec,'component','Sperp', 'imagChk', False, 'hermit', False, 'autoEmin', True, 'Evect', np.linspace(energy_range[num][0],energy_range[num][1],img_shape[0]+1))
                csbSpec = m.sw_instrument(csbSpec, dE=dE) # 1 parameters
                #res = np.flipud(csbSpec['swConv'].real)
                res = csbSpec['swConv'].real
            except:
                res = np.ones(img_shape, dtype=float)*0.1
        else:
            csbSpec = csb.spinwave(QQ,'formfact',True)
            csbSpec = m.sw_neutron(csbSpec)
            #csbSpec = m.sw_egrid(csbSpec,'component','Sperp', 'imagChk', False)
            csbSpec = m.sw_egrid(csbSpec,'component','Sperp', 'autoEmin', True, 'Evect', np.linspace(energy_range[num][0],energy_range[num][1],img_shape[0]+1))
            csbSpec = m.sw_instrument(csbSpec, dE=dE) # 1 parameters
            #res = np.flipud(csbSpec['swConv'].real)
            res = csbSpec['swConv'].real
        results.append(res)
    
    ##---------------finishing to change matlab spinw function --> #output res#---------------##
    return results


# # Spinwave calculation function
class Spinwave:
    def __init__(self, dim=12, img_shape=img_shape, Q_paths=Q_paths, energy_range=energy_range):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.img_shape = img_shape
        self.Q_paths = Q_paths
        self.energy_range=energy_range
        self.bounds = np.stack([self.lb, self.ub])
                
    def __call__(self, X):
        assert len(X) == self.dim
        assert X.ndim == 1
        assert np.all(X <= self.ub) and np.all(X >= self.lb)
        
        os.chdir(root_path)
        
        ### loss function
        loss_func = torch.nn.L1Loss(reduction='sum')

        ### generate the sample matrix image
        if os.path.isfile(f'{root_path}para_results.txt'):
            iter_num = int(os.popen('cat '+root_path+'para_results.txt | wc -l').read())+1
            # with open(f'{self.root_path}/parameters/{self.output_file_name}', 'rb') as file:
            #     previous_state = pickle.load(file)
        else:
            iter_num = 1
            # previous_state = None
        
        # Redirect standard output and error to null device
        sys.stdout = open(os.devnull, 'w')
        #sys.stderr = open(os.devnull, 'w')
        ### Calculate target function
        res = func(X, img_shape=self.img_shape, energy_range=self.energy_range, Q_paths=self.Q_paths, fit=True)
        # Restore standard output and error
        sys.stdout = sys.__stdout__
        #sys.stderr = sys.__stderr__
        
        folder_init = int(os.popen(f'cat {root_path}dict_candi | wc -l').read())
        folder = str(folder_init)
        
        if not os.path.isdir(root_path+'previous/'):
            os.mkdir(root_path+'previous/')
        if not os.path.isdir(root_path+'previous/'+folder):
            os.mkdir(root_path+'previous/'+folder)
        
        Loss = []
        for QQ in range(len(self.Q_paths)):
            
            io = IO(root_path, QQ)
            [LX, LY] = io.read_SW(kf_name=f'{QQ}_SW_stat_point.txt')   ### TBD
            LX_index = io.data_index_transformation(LX, 'expdata_2_simuindex')
            
            base_mat = np.zeros(self.img_shape, dtype=float)

            for count,index in enumerate(LX_index):
                map_index = [int(i) for i in index]
                base_mat[tuple(map_index)] = LY[count]
            
            base_mat = base_mat / max(np.max(base_mat), 0.000001)
            
            fitted_res = res[QQ] / max(np.max(res[QQ]), 0.000001)
            fbase_mat = np.zeros(img_shape, dtype=float)
            
            for count,index in enumerate(LX_index):
                map_index = [int(i) for i in index]
                fbase_mat[tuple(map_index)] = fitted_res[tuple(map_index)]
            fbase_mat = fbase_mat/max(np.max(fbase_mat), 0.000001)
            
            loss = (loss_func(torch.tensor(fbase_mat).to(dtype=dtype, device=device), torch.tensor(base_mat).to(dtype=dtype, device=device))).detach().cpu().numpy()/len(LY)
            
            if np.isnan(loss):
                loss = 1
                
            Loss.append(loss)
            
        with open(root_path+'dict_candi','a+') as f:
            f.write(folder+'   '+'_'.join([str(num) for num in X])+'   '+'  '.join([str(loss) for loss in Loss])+'\n')
        
        output = np.sum(np.array(Loss))
        
        with open(root_path+'previous/'+folder+'/out.pickle', 'wb') as f:
            pickle.dump([X, res], f)

        return output


#------------------------------------------------------- ABO fnctions and assosiated functions ------------------------------------------------------#

###### set the bound limitation for bayesian ######

### [-1.9034, -3.3792, -1.6698, -0.093345, -0.089593, 0, 0.36648, -0.29315, 0.0013, -0.0076, 0.31, 3]

### [J1, J2, J3, J4, J5, J6, J7, J8, Aniso1, Aniso2, DM1, dE]
# lb = np.array([-4, -7, -4, -0.5, -0.5,  0,   0, -0.5,   0, -0.5,   0,  0])
# ub = np.array([ 0,  0,  0,    0,    0,  2, 0.5,    0, 0.5,    0, 0.5,  6])

## [J1, J2, J3, J4, J5, J7, J8, Aniso1, Aniso2, DM1, dE]
lb = np.array([-4, -7, -4, -0.5, -0.5,   0, -0.5,   0,  0])
ub = np.array([ 0,  0,  0,    0,    0, 0.5,    0, 0.5,  6])

fun_q = Spinwave(dim=len(lb), img_shape=img_shape, energy_range=energy_range, Q_paths=Q_paths)
fun_q.lb = np.array(lb)
fun_q.ub = np.array(ub)
dim_q = fun_q.dim
print(fun_q.bounds)

def cos_sim(A,B):
    cosine = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return cosine

def candidate_selection(X, fX, max_size = None):

    ind_best = np.argmin(fX)
    ind_list = np.argsort(fX, axis=0).squeeze()
    
    sort_fX = fX[ind_list]
    sort_X = X[ind_list]

    screened_number = np.sum((sort_fX<fX[ind_best]*2)!=0)
    
    cos_similarity = [cos_sim(sort_X[0], i) for i in sort_X]
    select_fX = [sort_fX[0]]
    select_X = [sort_X[0]]
    select_index = [ind_list[0]]
    
    for i in range(1, screened_number):
        cosS = np.array([cos_sim(sort_X[i], j) for j in select_X])
        print((cosS<=0.96).all())
        if (cosS<=0.96).all():
            select_fX.append(sort_fX[i])
            select_X.append(sort_X[i])
            select_index.append(ind_list[i])        
    
    return np.array(select_X)[:max_size], np.array(select_fX)[:max_size], np.array(select_index)[:max_size]


###### BO_search Quanty parameters ######
def BO_search_SW_Turbo(fun_q, n_trust_regions=4,  batch_size=4, max_evals=1000):
    
    #n_init = n_trust_regions*batch_size
    n_init = 4

    turbo_m = TurboM(
        f=fun_q,  # Handle to objective function
        lb=fun_q.lb,  # Numpy array specifying lower bounds
        ub=fun_q.ub,  # Numpy array specifying upper bounds
        n_init=n_init,  # Number of initial bounds from an Symmetric Latin hypercube design
        max_evals=max_evals,  # Maximum number of evaluations
        n_trust_regions=n_trust_regions,  # Number of trust regions
        batch_size=batch_size,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=1000,  # When we switch from Cholesky to Lanczos
        n_training_steps=100,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
    
    turbo_m.optimize()

    X = turbo_m.X  # Evaluated points
    fX = turbo_m.fX  # Observed values
    
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]
    
    resX, resfX, resindex = candidate_selection(X, fX)
    
    return {'X_best': torch.tensor(x_best, device=device, dtype=dtype), 'Best_value': torch.tensor(f_best[0], device=device, dtype=dtype), 'cand_models': resX, 'cand_index': resindex}


### function for candidate BO for SW
def find_peaks(matrix, Xs, neighbor_size = 2):
    rows, cols = matrix.shape
    peaks = []  # List to store the peaks with their metrics

    # Determine the neighborhood size based on the density of Xs
    total_points = len(Xs)
    neighborhood_size = min(12, int(total_points / 12)+1)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = matrix[i, j]
            neighbor_mat = matrix[max(0, i-neighbor_size):i+neighbor_size+1, max(0, j-neighbor_size):j+neighbor_size+1]
            center_mat = np.ones(neighbor_mat.shape)*center
            
            if np.all(center_mat >= neighbor_mat):
                
                # Calculate the density of sampled points around the peak
                x_min, x_max = max(0, i - neighborhood_size), min(rows, i + neighborhood_size + 1)
                y_min, y_max = max(0, j - neighborhood_size), min(cols, j + neighborhood_size + 1)
                density = sum(1 for x, y in Xs if x_min <= x < x_max and y_min <= y < y_max)+1
                
                # Evaluation metric: peak height minus a factor of density
                metric = center / np.sqrt(density)
                peaks.append(((i, j), metric, center))
                #peaks.append(((i, j), center， density))
    
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    #num_peaks = min(len(peaks), 32)
    num_peaks = len(peaks)
    top_peaks = peaks[:num_peaks]
    
    pos = np.array([i[0] for i in top_peaks])
    peak = np.array([i[2] for i in top_peaks])
    pos = pos[peak > np.max(peak)*0.01]
    
    return pos


###### BO_search SW points' positions ######
def BO_search_SW(Candidate_S, minhpar=1, option='FNGP', Q=0, exp_img_shape=exp_img_shape, batch=8):
    
    ###### get Quanty parameters and results ######
    
    io = IO(root_path, Q)
    folder = str(int(os.popen('cat '+root_path+'para_results.txt | wc -l').read()))
    print('\nNum_sample_iterations:', folder)
    
    ### get fitted SW
    ### TBD: we can either read best fitting or average fitting, check the performance later. But set to mean value now.
    #_, _fY = io.read_output('parameters/'+folder)
    kfY = io.read_output_KF('parameters/'+folder)
    _fY, _pfY = kfY[Q]
    #fY = _fY / np.max(_fY)
    
    ### get previous sampled SW points
    #[_X, _Y] = io.read_SW(kf_name=f'{Q}_SW_stat_point.txt')
    [_X, _Y] = io.read_SW(kf_name=f'{Q}_SW_point.txt')
    LX_index = io.data_index_transformation(_X, 'expdata_2_simuindex')
    LX_index_exp = io.data_index_transformation(_X, 'expdata_2_expindex')
    #already_exist_sample = [tuple(i) for i in _X]
    already_exist_sample = [tuple(i) for i in LX_index_exp]

    ################################################ Bayesian Sampling ##################################################

    _new_fY = []
    for count,index in enumerate(LX_index):
        map_index = [int(i) for i in index]
        _new_fY.append(_fY[tuple(map_index)])
    _new_fY = np.array(_new_fY)

    _Y = _Y/np.max(_Y)
    new_fY = _new_fY/max(np.max(_new_fY), 0.000001)
    fY = _fY/max(np.max(_new_fY), 0.000001)
    _npfY = _pfY/max(np.max(_new_fY), 0.000001)
    
    ################################################ Image Bayesian ###################################################
    ### Conduct the last step bayesian sampling to find possibly high value index

    ### Train Gaussian for original Image space
    ori_bounds = torch.tensor([[0,0], exp_img_shape[Q]], device=device, dtype=torch.float)
    dim = ori_bounds.shape[-1]
    
    tensor_LX_index = torch.tensor(LX_index_exp, device=device, dtype=dtype)
    X_ori = normalize(tensor_LX_index, ori_bounds)

    Y_diff = torch.tensor(_Y-new_fY, dtype=dtype, device=device).unsqueeze(-1)
    
    if len(Y_diff) == 1:
        Y_train_diff = Y_diff - Y_diff.mean()
    else:
        Y_train_diff = (Y_diff - Y_diff.mean()) / Y_diff.std()
    Yvar_diff = torch.abs(Y_train_diff) * 0.1

    ## SpectralDeltaKernel is accurate but may collapse when data are accumulate at some region
    ## covar_module = ScaleKernel(SpectralDeltaKernel(num_dims=dim, num_deltas=2))
    ## MaternKernel is stable but accuracy is relative lower
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.10, 4.0)))
    
    model_G = FixedNoiseGP(X_ori, Y_train_diff, Yvar_diff, covar_module=covar_module)
    mll_G = ExactMarginalLogLikelihood(model_G.likelihood, model_G)
    fit_gpytorch_model(mll_G)
    
    model_G_pred = model_G.posterior(Candidate_S)
    res = model_G_pred.mean*Y_diff.std()+Y_diff.mean()
    var = torch.sqrt(model_G_pred.variance)*Y_diff.std()
    
    res = res.detach().cpu().numpy().squeeze()
    var = var.detach().cpu().numpy().squeeze()
    #varhpar = min(np.mean(np.abs(_Y-new_fY))/np.mean(_Y), minhpar)
    varhpar = np.mean(np.abs(_Y-new_fY))/np.mean(_Y)
    print(f'varhpar: {varhpar}')
    # print(f'imgspace values: mean: {np.mean(res)}, hpar: {hpar}, hpar_var: {np.mean(hpar*var)}')
    
    fsample_base = np.zeros(img_shape, dtype=float)
    fsample_resi = np.zeros(img_shape, dtype=float)
    fsample_resi_var = np.zeros(img_shape, dtype=float)
    for count,index in enumerate(unnormalize(Candidate_S, ori_bounds)):
        map_index = [int(i) for i in index]
        #ratio = var[count] / (_npfY[tuple(map_index)]+var[count])
        ratio = _npfY[tuple(map_index)] / (_npfY[tuple(map_index)]+var[count])
        fsample_base[tuple(map_index)] = fY[tuple(map_index)] + minhpar*(ratio*(res[count] + varhpar*var[count])
        fsample_resi[tuple(map_index)] = res[count]
        fsample_resi_var[tuple(map_index)] = var[count]
        
    X_all = find_peaks(fsample_base, _X, neighbor_size = int(np.ceil(min(exp_img_shape[Q])/80)))
    Next_sample_index = []
    for index in X_all:
        select_index = tuple(index)
        if select_index not in already_exist_sample:
            Next_sample_index.append(select_index)
    
    Next_sample_index = Next_sample_index[:min(batch, len(Next_sample_index))]

    # pred_resi = []   
    # for index in already_exist_sample:
    #     map_index = [int(i) for i in index]
    #     pred_resi.append(fsample_resi[tuple(map_index)])
        
    # for index in Next_sample_index:
    #     map_index = [int(i) for i in index]
    #     pred_resi.append(fsample_resi[tuple(map_index)])
        
    Next_sample = io.data_index_transformation(Next_sample_index, 'expindex_2_expdata')
    
    ################################################ Finish OriimgBaye ################################################
    
    print('\n--------------------------------------------') 
    print(f"Sampled numbers of path_{Q}: {len(Next_sample_index)}")
    print(f"New samples_index: {Next_sample_index}")
    print(f"New samples: {Next_sample}\n\n\n")
        
    return {'X_best': Next_sample, 'Pred_residual': fsample_resi, 'Pred_resivar': fsample_resi_var}

def simple_KF(root_path, Q, predX_residual, predX_resivar, num_new):

    io = IO(root_path, Q)
    folder = str(int(os.popen('cat '+root_path+'para_results.txt | wc -l').read()))

    os.system(f'mv {Q}_SW_stat_point.txt parameters/{str(int(folder))}/{Q}_SW_stat_point.txt')
    
    ### get fitted SW and its variance
    kfY = io.read_output_KF('parameters/'+folder)
    _fY, _pfY = kfY[Q]
    #fY = _fY / np.max(_fY)
    
    ### get previous sampled SW points
    [_X, _Y] = io.read_SW(kf_name=f'{Q}_SW_point.txt')
    LX_index = io.data_index_transformation(_X, 'expdata_2_simuindex')
    LX_index_exp = io.data_index_transformation(_X, 'expdata_2_expindex')
    already_exist_sample = [tuple(i) for i in LX_index_exp]

    _new_fY = []
    _new_pfY = []
    for count,index in enumerate(LX_index):
        map_index = [int(i) for i in index]
        _new_fY.append(_fY[tuple(map_index)])
        _new_pfY.append(_pfY[tuple(map_index)])
        
    _new_fY = np.array(_new_fY)
    _new_pfY = np.array(_new_pfY)
    _Y = _Y/np.max(_Y)
    new_fY = _new_fY/max(np.max(_new_fY), 0.000001)
    new_pfY = _new_pfY/max(np.max(_new_fY), 0.000001)
    fY = _fY/max(np.max(_new_fY), 0.000001)
    expr_resi = _Y-new_fY
    
    predX_res = []
    predX_resval = []
    for index in already_exist_sample:
        map_index = [int(i) for i in index]
        predX_res.append(predX_residual[tuple(map_index)])
        predX_resval.append(predX_resivar[tuple(map_index)])

    predX_res = np.array(predX_res)
    predX_resval = np.array(predX_resval)
    
    print('expr_predX: ', expr_resi.shape, predX_res.shape, predX_resval.shape)
    print((expr_resi[-num_new:], predX_res[-num_new:], predX_resval[-num_new:]))

    ## KF residual
    trust_exp_noise_ratio = cos_sim(expr_resi[-num_new:], predX_res[-num_new:])
    print(trust_exp_noise_ratio)
    residual = predX_res + (1-trust_exp_noise_ratio)*(expr_resi-predX_res)
    final_Y = new_fY + (new_pfY/(np.abs(predX_resval)+new_pfY))*residual

    ## Normal residual
    # final_Y = new_fY + (new_pfY/(np.abs(predX_resval)+new_pfY))*expr_resi

    
    # final_Y[-num_new:] = _Y[-num_new:]

    with open(f'{Q}_SW_stat_point.txt', 'w') as f:
        for num,j in enumerate(_X):
            f.write(str(j[0])+'   '+str(j[1])+'   '+str(final_Y[num])+'   '+str(folder)+'\n')

#--------------------------------------------------------- End of the Main functions --------------------------------------------------------#



#--------------------------------------------------------- Initialization and restart --------------------------------------------------------#
def get_initial_points(dim, n_pts):
    #sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def generate_candidate(root_path):

    Sampling_Candidate = []
    for exp_num in range(len(Expr.keys())):

        sample_matrix = []
        sample_x, sample_y = np.arange(exp_img_shape[exp_num][0]), np.arange(exp_img_shape[exp_num][1])
        for i in range(len(sample_x)):
            for j in range(len(sample_y)):
                sample_matrix.append([sample_x[i], sample_y[j]])
        ori_bounds = torch.tensor([[0,0], exp_img_shape[exp_num]], device=device, dtype=torch.float)
        Candidate = normalize(torch.tensor(np.array(sample_matrix)).to(dtype=dtype, device=device), ori_bounds)
        Sampling_Candidate.append([Candidate, ori_bounds])

    return Sampling_Candidate

def generate_initial_grid(interval, img_shape):
    
    rim_x = (img_shape[0]-int(img_shape[0]/interval)*interval)/2
    rim_y = (img_shape[1]-int(img_shape[1]/interval)*interval)/2
    x = np.arange(rim_x+int(interval/2), img_shape[0], interval)
    y = np.arange(rim_y+int(interval/2), img_shape[1], interval)
    
    xv, yv = np.meshgrid(x, y)
    grid_points = np.vstack([xv.ravel(), yv.ravel()]).T
    
    return grid_points


#---------------------------------------- Run the code ---------------------------------------#
os.chdir(root_path)
Candidate_S = generate_candidate(root_path)

initialize = True
for Q in range(len(Q_paths)):
    if not os.path.isfile(f'{Q}_SW_point.txt'):
        initialize = False  

### Experiment state judgement
##--------------------------------------##
if os.path.isfile('para_results.txt'):
    restart = True
else:
    restart = False

if initialize is False and restart is True:
    new_search = True
else:
    new_search = False
##--------------------------------------##


if initialize is False:    
    for Q in range(len(Q_paths)):
        io = IO(root_path, Q)
        _X_data = generate_initial_grid(interval=15, img_shape=exp_img_shape[Q])
        X_data = io.data_index_transformation(_X_data, 'expindex_2_expdata')
        print('initialization_data: ', X_data)
        X = X_data
        _ = get_SW(X, Q)
        os.system(f'cp {Q}_SW_point.txt {Q}_SW_stat_point.txt')
        
    initialize = True

io = IO(root_path, 0)

if restart is False or new_search is True:

    result = BO_search_SW_Turbo(fun_q, n_trust_regions=12, batch_size=3, max_evals=max_evals)
    #result = BO_search_quanty_BWO(fun_q, lb=fun_q.lb, ub=fun_q.ub, dim=fun_q.dim, SearchAgents_no=16, Max_iteration=200)
    
    stop_value = result['Best_value'].cpu().numpy()

    if not os.path.isdir(root_path+'nn_files/'):
        os.mkdir(root_path+'nn_files/')
    nn_num = int(os.popen('ls '+root_path+'nn_files | wc -l').read())+1
    if not os.path.isdir(root_path+'nn_files/'+str(nn_num)):
        os.mkdir(root_path+'nn_files/'+str(nn_num))
    os.system('mv dict_candi previous nn_files/'+str(nn_num))
    io.database_convert(i=nn_num, index=result['cand_index'])
    os.system('mv database_nn.pickle nn_files/'+str(nn_num))
    os.system(f'rm -r nn_files/{str(nn_num)}/previous')
    os.mkdir('previous')
 
    para_init = int(os.popen('cat '+root_path+'para_results.txt | wc -l').read())+1
    with open('para_results.txt', 'a') as f:
        f.write(str(para_init)+'   '+'_'.join([str(i.item()) for i in result['X_best']])+'   '+str(result['Best_value'].cpu().numpy())+'\n')
    read_SW_para(root_path=root_path, file='para_results.txt')
    os.system('mv stat_model.pickle parameters/'+str(para_init))
    
    print('finish_reading')
    
    
n = int(os.popen('cat '+root_path+'SW_point.txt | wc -l').read())

while n < 1000:
    
    print('\n n: ', n)
    
    para_num = int(os.popen('cat '+root_path+'para_results.txt | wc -l').read())+1
    print('parameter_num: ', para_num)

    for Q in range(len(Q_paths)):
        x_point = BO_search_SW(Candidate_S[Q][0], option='FNGP', Q=Q)
        #os.system('cp fitted_sample.pickle parameters/'+str(para_num)+'/')
        _ = get_SW(x_point['X_best'], Q)
        simple_KF(root_path, Q, x_point['Pred_residual'], x_point['Pred_resivar'], len(x_point['X_best']))
    
    
    
    result = BO_search_SW_Turbo(fun_q, n_trust_regions=12, batch_size=3, max_evals=max_evals)
    #result = BO_search_quanty_BWO(fun_q, lb=fun_q.lb, ub=fun_q.ub, dim=fun_q.dim, SearchAgents_no=16, Max_iteration=300)
    
    if not os.path.isdir(root_path+'nn_files/'):
        os.mkdir(root_path+'nn_files/')
    nn_num = int(os.popen('ls '+root_path+'nn_files | wc -l').read())+1
    if not os.path.isdir(root_path+'nn_files/'+str(nn_num)):
        os.mkdir(root_path+'nn_files/'+str(nn_num)) 
    os.system('mv dict_candi previous nn_files/'+str(nn_num))
    io.database_convert(i=nn_num, index=result['cand_index'])
    os.system('mv database_nn.pickle nn_files/'+str(nn_num))
    os.system(f'rm -r nn_files/{str(nn_num)}/previous')
    os.mkdir('previous')
    stop_value = result['Best_value'].cpu().numpy()
    print('stop_value_accurate: ', stop_value)

    para_init = int(os.popen('cat '+root_path+'para_results.txt | wc -l').read())+1
    with open('para_results.txt', 'a') as f:
        f.write(str(para_init)+'   '+'_'.join([str(i.item()) for i in result['X_best']])+'   '+str(result['Best_value'].cpu().numpy())+'   No_restart\n')
    read_SW_para(root_path=root_path, file='para_results.txt')
    os.system('mv stat_model.pickle parameters/'+str(para_init))
    
    n += 1
