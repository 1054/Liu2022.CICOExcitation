#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os, sys, re, copy, time, datetime
import numpy as np
import astropy.units as u
from astropy.table import Table
from collections import OrderedDict
sys.path.insert(1, os.path.join(os.path.dirname(os.getcwd()), 'utils'))
dzliu_calc_path = os.path.expanduser('~/Cloud/GitLab/COExcitation/tmp/calc_dzliu')
LTE_calc_path = os.path.expanduser('~/Cloud/GitLab/COExcitation/tmp/calc_LTE')


# In[12]:


import moldata
import importlib
importlib.reload(moldata)

CO = moldata.CO()
print(CO)

CI = moldata.CI()
print(CI)


# In[13]:


# User defined global parameters

list_of_N_CO = 10**np.arange(17.0, 20.0, 0.25)
list_of_X_CICO = np.arange(0.05, 1.0+0.05, 0.05)
list_of_n_H2 = [1e2, 1e3, 1e4, 1e5, 1e6]
list_of_T_kin = [25., 50., 100.]
line_width = 5.

out_table_name = 'out_table_DZLIU_R_CI10CO21_sim_XCICO_dv_%gkms'%(line_width)


# In[14]:


# Define function to compute non-LTE solutions for one (T_kin, N_CO/line_width, n_H2, X_CICO)

def run_for_one_input_set(var_dict):
    # 
    global dzliu_calc_path
    global LTE_calc_path
    # 
    # read input variables
    CO = var_dict['CO'] # the moldata class object
    CI = var_dict['CI'] # the moldata class object
    N_CO = var_dict['N_CO'] # species_column_density of CO
    N_CI = var_dict['N_CI'] # species_column_density of CI
    X_CICO = N_CI/N_CO
    d_V = var_dict['d_V']
    T_kin = var_dict['T_kin']
    n_H2 = var_dict['n_H2']
    collision_partner_densities = var_dict['collision_partner_densities']
    # 
    # prepare working directory for LTE calculation temporary files
    T_kin_str = '%.0f'%(T_kin.to(u.K).value)
    N_CO_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', '%.2e'%(N_CO.to(u.cm**(-2)).value))
    n_H2_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', '%.1e'%(n_H2.to(u.cm**(-3)).value))
    d_V_str = '%.1f'%(d_V.to(u.km/u.s).value)
    calc_name = 'calc_T_kin_%s_N_CO_%s_n_H2_%s_d_V_%s'%(T_kin_str, N_CO_str, n_H2_str, d_V_str)
    LTE_working_dir = os.path.join(LTE_calc_path, calc_name)
    if not os.path.isdir(LTE_working_dir):
        os.makedirs(LTE_working_dir)
    if not os.path.isdir(LTE_working_dir):
        raise Exception('Error! Could not create dir: ' + LTE_working_dir)
    # 
    # prepare working directory for dzliu Non-LTE calculation temporary files
    NonLTE_working_dir = os.path.join(dzliu_calc_path, calc_name)
    if not os.path.isdir(NonLTE_working_dir):
        os.makedirs(NonLTE_working_dir)
    if not os.path.isdir(NonLTE_working_dir):
        raise Exception('Error! Could not create dir: ' + NonLTE_working_dir)
    
    # prepare CI file name, which includes a suffix indicating the CI/CO intrinsic abundance ratio
    X_CICO_str = '%.2f'%(X_CICO)
    CI_filename = 'ci_X_CICO_%s'%(X_CICO_str)
    
    # check CI LTE table file, read it if exists, otherwise do the computation
    out_table_file_CI_LTE = os.path.join(LTE_working_dir, CI_filename + '.csv')
    did_CI_LTE = False
    if os.path.isfile(out_table_file_CI_LTE):
        out_table_CI_LTE = Table.read(out_table_file_CI_LTE, format='csv')
    else:
        # use dzliu ../utils/moldata.py class to compute LTE for CI
        CI.evaluate_level_populations(species_column_density = N_CI, 
                                      line_width = d_V, 
                                      T_kin = T_kin, 
                                      collision_partner_densities = collision_partner_densities, 
                                      verbose = False, 
                                      silent = True, 
                                      LTE = True)
        out_table_CI_LTE = CI.get_solved_transition_properties()
        out_table_CI_LTE.write(out_table_file_CI_LTE, format='csv')
        did_CI_LTE = True
    
    # check CO LTE table file, read it if exists, otherwise do the computation
    out_table_file_CO_LTE = os.path.join(LTE_working_dir, 'co.csv')
    did_CO_LTE = False
    if os.path.isfile(out_table_file_CO_LTE):
        out_table_CO_LTE = Table.read(out_table_file_CO_LTE, format='csv')
    else:
        # use dzliu ../utils/moldata.py class to compute LTE for CO
        CO.evaluate_level_populations(species_column_density = N_CO, 
                                      line_width = d_V, 
                                      T_kin = T_kin, 
                                      collision_partner_densities = collision_partner_densities, 
                                      verbose = False, 
                                      silent = True, 
                                      LTE = True)
        out_table_CO_LTE = CO.get_solved_transition_properties()
        out_table_CO_LTE.write(out_table_file_CO_LTE, format='csv')
        did_CO_LTE = True
    
    # now compute NonLTE
    
    # check CI NonLTE table file, read it if exists, otherwise do the computation
    out_table_file_CI_NonLTE = os.path.join(NonLTE_working_dir, CI_filename + '.csv')
    if os.path.isfile(out_table_file_CI_NonLTE):
        out_table_CI_NonLTE = Table.read(out_table_file_CI_NonLTE, format='csv')
    else:
        # use dzliu ../utils/moldata.py class to compute LTE for CI
        if os.path.isfile(out_table_file_CI_NonLTE + '.warning.not.converged'):
            os.remove(out_table_file_CI_NonLTE + '.warning.not.converged')
        if not did_CI_LTE:
            CI.evaluate_level_populations(species_column_density = N_CI, 
                                          line_width = line_width, 
                                          T_kin = T_kin, 
                                          collision_partner_densities = collision_partner_densities, 
                                          verbose = False, 
                                          silent = True, 
                                          LTE = True)
        #out_table_CI_LTE = CI.get_solved_transition_properties()
        CI.solve_rate_matrix_iteratively(verbose = False, silent = True)
        if (CI.converged_iterations<0):
            os.system('touch "%s"'%(out_table_file_CI_NonLTE + '.warning.not.converged'))
        out_table_CI_NonLTE = CI.get_solved_transition_properties()
        out_table_CI_NonLTE.write(out_table_file_CI_NonLTE, format='csv')
    
    # check CO NonLTE table file, read it if exists, otherwise do the computation
    out_table_file_CO_NonLTE = os.path.join(NonLTE_working_dir, 'co.csv')
    if os.path.isfile(out_table_file_CO_NonLTE):
        out_table_CO_NonLTE = Table.read(out_table_file_CO_NonLTE, format='csv')
    else:
        # use dzliu ../utils/moldata.py class to compute NonLTE for CO
        if os.path.isfile(out_table_file_CO_NonLTE + '.warning.not.converged'):
            os.remove(out_table_file_CO_NonLTE + '.warning.not.converged')
        if not did_CO_LTE:
            CO.evaluate_level_populations(species_column_density = N_CO, 
                                          line_width = line_width, 
                                          T_kin = T_kin, 
                                          collision_partner_densities = collision_partner_densities, 
                                          verbose = False, 
                                          silent = True, 
                                          LTE = True)
        #out_table_CO_LTE = CO.get_solved_transition_properties()
        CO.solve_rate_matrix_iteratively(verbose = False, silent = True)
        if (CO.converged_iterations<0):
            os.system('touch "%s"'%(out_table_file_CO_NonLTE + '.warning.not.converged'))
        out_table_CO_NonLTE = CO.get_solved_transition_properties()
        out_table_CO_NonLTE.write(out_table_file_CO_NonLTE, format='csv')
    
    # prepare output result dict
    res_dict = OrderedDict()
    res_dict['N_CO'] = N_CO
    res_dict['N_CI'] = N_CI
    res_dict['X_CICO'] = N_CI/N_CO
    res_dict['T_kin'] = T_kin
    res_dict['n_H2'] = n_H2
    res_dict['d_V'] = d_V
    
    res_dict['Flux_Kkms_CO10_NonLTE'] = out_table_CO_NonLTE['Flux_Kkms'][0]
    res_dict['Flux_Kkms_CO10_LTE'] = out_table_CO_LTE['Flux_Kkms'][0]
    res_dict['Flux_Kkms_CO21_NonLTE'] = out_table_CO_NonLTE['Flux_Kkms'][1]
    res_dict['Flux_Kkms_CO21_LTE'] = out_table_CO_LTE['Flux_Kkms'][1]
    res_dict['Flux_Kkms_CI10_NonLTE'] = out_table_CI_NonLTE['Flux_Kkms'][0]
    res_dict['Flux_Kkms_CI10_LTE'] = out_table_CI_LTE['Flux_Kkms'][0]
    res_dict['Flux_Kkms_CI21_NonLTE'] = out_table_CI_NonLTE['Flux_Kkms'][1]
    res_dict['Flux_Kkms_CI21_LTE'] = out_table_CI_LTE['Flux_Kkms'][1]
    
    res_dict['tau_0_CO10_NonLTE'] = out_table_CO_NonLTE['tau_0'][0]
    res_dict['tau_0_CO10_LTE'] = out_table_CO_LTE['tau_0'][0]
    res_dict['tau_0_CO21_NonLTE'] = out_table_CO_NonLTE['tau_0'][1]
    res_dict['tau_0_CO21_LTE'] = out_table_CO_LTE['tau_0'][1]
    res_dict['tau_0_CI10_NonLTE'] = out_table_CI_NonLTE['tau_0'][0]
    res_dict['tau_0_CI10_LTE'] = out_table_CI_LTE['tau_0'][0]
    res_dict['tau_0_CI21_NonLTE'] = out_table_CI_NonLTE['tau_0'][1]
    res_dict['tau_0_CI21_LTE'] = out_table_CI_LTE['tau_0'][1]
    
    res_dict['Pop_u_CO10_NonLTE'] = out_table_CO_NonLTE['Pop_u'][0]
    res_dict['Pop_u_CO10_LTE'] = out_table_CO_LTE['Pop_u'][0]
    res_dict['Pop_u_CO21_NonLTE'] = out_table_CO_NonLTE['Pop_u'][1]
    res_dict['Pop_u_CO21_LTE'] = out_table_CO_LTE['Pop_u'][1]
    res_dict['Pop_u_CI10_NonLTE'] = out_table_CI_NonLTE['Pop_u'][0]
    res_dict['Pop_u_CI10_LTE'] = out_table_CI_LTE['Pop_u'][0]
    res_dict['Pop_u_CI21_NonLTE'] = out_table_CI_NonLTE['Pop_u'][1]
    res_dict['Pop_u_CI21_LTE'] = out_table_CI_LTE['Pop_u'][1]
    
    res_dict['Pop_l_CO10_NonLTE'] = out_table_CO_NonLTE['Pop_l'][0]
    res_dict['Pop_l_CO10_LTE'] = out_table_CO_LTE['Pop_l'][0]
    res_dict['Pop_l_CO21_NonLTE'] = out_table_CO_NonLTE['Pop_l'][1]
    res_dict['Pop_l_CO21_LTE'] = out_table_CO_LTE['Pop_l'][1]
    res_dict['Pop_l_CI10_NonLTE'] = out_table_CI_NonLTE['Pop_l'][0]
    res_dict['Pop_l_CI10_LTE'] = out_table_CI_LTE['Pop_l'][0]
    res_dict['Pop_l_CI21_NonLTE'] = out_table_CI_NonLTE['Pop_l'][1]
    res_dict['Pop_l_CI21_LTE'] = out_table_CI_LTE['Pop_l'][1]
    
    res_dict['T_ex_CO10_NonLTE'] = out_table_CO_NonLTE['T_ex'][0]
    res_dict['T_ex_CO10_LTE'] = out_table_CO_LTE['T_ex'][0]
    res_dict['T_ex_CO21_NonLTE'] = out_table_CO_NonLTE['T_ex'][1]
    res_dict['T_ex_CO21_LTE'] = out_table_CO_LTE['T_ex'][1]
    res_dict['T_ex_CI10_NonLTE'] = out_table_CI_NonLTE['T_ex'][0]
    res_dict['T_ex_CI10_LTE'] = out_table_CI_LTE['T_ex'][0]
    res_dict['T_ex_CI21_NonLTE'] = out_table_CI_NonLTE['T_ex'][1]
    res_dict['T_ex_CI21_LTE'] = out_table_CI_LTE['T_ex'][1]
    
    try:
        res_dict['R_CI10CO21_NonLTE'] = out_table_CI_NonLTE['Flux_Kkms'][0] / out_table_CO_NonLTE['Flux_Kkms'][1]
    except:
        res_dict['R_CI10CO21_NonLTE'] = np.nan
    try:
        res_dict['R_CI10CO21_LTE'] = out_table_CI_LTE['Flux_Kkms'][0] / out_table_CO_LTE['Flux_Kkms'][1]
    except:
        res_dict['R_CI10CO21_LTE'] = np.nan
    
    # return the result dict
    return res_dict

# test
res_dict = None
if True:
    res_dict = run_for_one_input_set({
                'CO': CO, 
                'CI': CI, 
                'T_kin': list_of_T_kin[0] * u.K, 
                'N_CO': list_of_N_CO[0] * u.cm**(-2), 
                'N_CI': list_of_N_CO[0] * list_of_X_CICO[0] * u.cm**(-2), 
                'X_CICO': list_of_X_CICO[0], 
                'n_H2': list_of_n_H2[0] * u.cm**(-3), 
                'd_V': line_width * u.km/u.s, 
                'collision_partner_densities': {'ortho-H2': 0.75 * list_of_n_H2[0] * u.cm**(-3), 'para-H2': 0.25 * list_of_n_H2[0] * u.cm**(-3)}, 
            }
        )
res_dict


# In[ ]:


# Run the calculation loop

print('datetime.datetime.now()', datetime.datetime.now())

from multiprocessing import Pool

# Loop
var_dict = {'CO': CO, 'CI': CI}
out_dict = None
mproc_list = []
N_CPU = 50
with Pool(N_CPU) as mproc_pool:
    for idx_X_CICO, X_CICO in enumerate(list_of_X_CICO):
        for idx_N_CO, N_CO in enumerate(list_of_N_CO):
            for idx_n_H2, n_H2 in enumerate(list_of_n_H2):
                for idx_T_kin, T_kin in enumerate(list_of_T_kin):
                    print('Parameters: (T_kin, n_H2, N_CO, X_CICO) = (%5.1f, %5.2f, %4.1f, %4.2f) | %02d/%02d, %02d/%02d, %02d/%02d, %02d/%02d |'%(
                        T_kin, 
                        np.log10(n_H2), 
                        np.log10(N_CO), 
                        X_CICO, 
                        idx_T_kin+1, len(list_of_T_kin), 
                        idx_n_H2+1, len(list_of_n_H2), 
                        idx_N_CO+1, len(list_of_N_CO), 
                        idx_X_CICO+1, len(list_of_X_CICO), 
                    ))
                    var_dict['N_CO'] = N_CO * u.cm**(-2)
                    var_dict['N_CI'] = X_CICO * N_CO * u.cm**(-2) # CI/CO intrinsic abundance ratio
                    var_dict['X_CICO'] = X_CICO
                    var_dict['d_V'] = line_width * u.km/u.s
                    var_dict['T_kin'] = T_kin * u.K
                    var_dict['n_H2'] = n_H2 * u.cm**(-3)
                    var_dict['collision_partner_densities'] = {'ortho-H2': 0.75 * n_H2 * u.cm**(-3), 'para-H2': 0.25 * n_H2 * u.cm**(-3)}
                    
                    #res_dict = run_for_one_input_set(var_dict)
                    
                    var_dict2 = copy.deepcopy(var_dict)
                    tmp_proc = mproc_pool.apply_async(run_for_one_input_set, args=(var_dict2, ))
                    mproc_list.append(tmp_proc)
					
                    time.sleep(0.5)

    for tmp_proc in mproc_list:
        res_dict = tmp_proc.get()
        if out_dict is None:
            out_dict = OrderedDict()
            for key in res_dict:
                out_dict[key] = [res_dict[key]]
        else:
            for key in res_dict:
                out_dict[key].append(res_dict[key])
    
    mproc_pool.close()
    mproc_pool.join()
        
print('datetime.datetime.now()', datetime.datetime.now())

out_table = Table(out_dict)
out_table


# In[ ]:


# Write table to disk

out_table.write(out_table_name + '.csv', format='csv', overwrite=True)
out_table.write(out_table_name + '.dat', format='ascii.fixed_width', delimiter='  ', bookend=True, overwrite=True)
print('Output to "%s"'%(out_table_name + '.csv'))
print('Output to "%s"'%(out_table_name + '.dat'))


# In[ ]:




