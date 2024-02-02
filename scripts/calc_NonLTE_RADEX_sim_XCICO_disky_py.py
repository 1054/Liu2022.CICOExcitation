#!/usr/bin/env python
# coding: utf-8

# In[268]:


import os, sys, re, time, datetime, shutil, copy, subprocess
import numpy as np
import astropy.units as u
from astropy.table import Table
from collections import OrderedDict
sys.path.insert(1, os.path.join(os.path.dirname(os.getcwd()), 'utils'))
radex_bin_path = os.path.join(os.path.dirname(os.getcwd()), 'bin', 'radex_LVG_linux')
radex_data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
radex_calc_path = os.path.join(os.path.dirname(os.getcwd()), 'tmp', 'calc_radex')
LTE_calc_path = os.path.join(os.path.dirname(os.getcwd()), 'tmp', 'calc_LTE')
CPU_cores = 50


# In[269]:


import moldata
import importlib
importlib.reload(moldata)

CO = moldata.CO()
print(CO)

CI = moldata.CI()
print(CI)


# In[270]:


# User defined global parameters

list_of_N_CO = 10**np.arange(17.0, 20.0+0.25, 0.25)
#list_of_X_CICO = np.arange(0.05, 3.00+0.05, 0.05)
list_of_X_CICO = np.concatenate([[0.05], np.arange(0.1, 1.0, 0.1), np.arange(1.0, 3.0+0.2, 0.2)])
#list_of_X_CICO = [0.1, 0.2, 1.0] # 20220309
list_of_n_H2 = [1.00e2, 3.16e2, 1.00e3, 3.16e3, 1.00e4, 3.16e4]
list_of_T_kin = [10., 15., 20., 25., 30., 40., 50., 60., 70., 80., 90., 100., 200., 300.]
# Note: for CI collison rates only exist for >= 10 K
list_of_d_V = [3., 4., 10., 15., 20., 25., 30., 40., 50., 60., 70., 90.]
line_width = 15. # for testing
#line_width = 4. #<TODO># manually set and run for: 3. 4. 10. 15. 
#line_width = 10. #<TODO><20220112># manually set and run for: 10.
#line_width = 15. #<TODO><20220112># manually set and run for: 15.
#line_width = 20. #<TODO><20220112># manually set and run for: 20.
#line_width = 25. #<TODO><20220112># manually set and run for: 25.
#line_width = 30. #<TODO><20220112># manually set and run for: 30.
#line_width = 40. #<TODO><20220112># manually set and run for: 40.
#line_width = 50. #<TODO><20220111># manually set and run for: 50.
#line_width = 60. #<TODO><20220112># manually set and run for: 60.
#line_width = 70. #<TODO><20220112># manually set and run for: 70.
#line_width = 90. #<TODO><20220112># manually set and run for: 90.


#list_of_n_H2 = [5e2]

#out_table_name = 'out_table_RADEX_R_CI10CO21_sim_XCICO_dv_%gkms'%(line_width)


# In[271]:


# Define function to compute LTE solutions for one (T_kin, N_CO/line_width, n_H2, X_CICO)

import sqliteutil
import importlib
importlib.reload(sqliteutil)

def calc_one_LTE_one_species(
        working_dir, species_name, T_CMB, T_kin, N_X, n_H2, d_V, overwrite, verbose, # must have these args
        LTE_calc_path = '', # then kwargs
        script_name = '', 
        moldata_object = None, 
    ):
    # 
    if LTE_calc_path != '':
        working_dir = os.path.join(LTE_calc_path, working_dir)
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    if script_name == '':
        script_name = species_name.lower()
    # 
    collision_partner_densities = {
        'ortho-H2': 0.75 * n_H2 * u.cm**(-3), 
        'para-H2': 0.25 * n_H2 * u.cm**(-3)
    }
    # 
    calc_csv_file = os.path.join(working_dir, script_name + '.csv')
    if verbose:
        print('calc_csv_file: ' + calc_csv_file + ' isfile? ' + str(os.path.isfile(calc_csv_file)))
    if not os.path.isfile(calc_csv_file) or overwrite:
        # use dzliu ../utils/moldata.py class to compute LTE for moldata_object
        moldata_object.evaluate_level_populations(
            species_column_density = N_X, 
            line_width = d_V, 
            T_kin = T_kin, 
            collision_partner_densities = collision_partner_densities, 
            verbose = verbose, 
            silent = True, 
            LTE = True)
        out_table_LTE = moldata_object.get_solved_transition_properties()
        for key in out_table_LTE.colnames:
            if isinstance(out_table_LTE[key][0], u.Quantity):
                out_table_LTE[key] = np.array([t.value for t in out_table_LTE[key]]).astype(float)
        out_table_LTE.write(calc_csv_file, format='csv', overwrite=overwrite)
    else:
        out_table_LTE = Table.read(calc_csv_file, format='csv')
    
    out_table_LTE['J_UP'] = np.array([t.replace('--',' ').split()[0] \
                                      for t in out_table_LTE['Line']]).astype(float)
    out_table_LTE['J_LOW'] = np.array([t.replace('--',' ').split()[-1] \
                                       for t in out_table_LTE['Line']]).astype(float)
    out_table_LTE['WAVE'] = 2.99792458e5/(out_table_LTE['Freq']/1e9)
    #for key in out_table_LTE.colnames:
    #    if isinstance(out_table_LTE[key][0], u.Quantity):
    #        out_table_LTE[key] = np.array([t.value for t in out_table_LTE[key]]).astype(float)
    # J_UP J_LOW E_u FREQ WAVE T_ex tau_9 T_RJ Pop_u Pop_l Flux_Kkms Flux_ergscmp2
    #out_table_LTE.remove_columns(['Line', 'I_nu_tau0', 'T_ant', 'Freq'])
    out_table_LTE.rename_column('Freq', 'FREQ')
    out_table_LTE.rename_column('T_ant', 'T_RJ')
    out_table_LTE = out_table_LTE[['J_UP', 'J_LOW', 'E_u', 'FREQ', 'WAVE', 
        'T_ex', 'tau_0', 'T_RJ', 'Pop_u', 'Pop_l', 'Flux_Kkms', 'Flux_ergscmp2']]
    
    out_dict_LTE = OrderedDict([(t,out_table_LTE[t].data.tolist()) for t in list(out_table_LTE.colnames)])
    if verbose:
        print('out_dict_LTE.keys() ' + str(out_dict_LTE.keys()))

    # cleanup
    shutil.rmtree(working_dir)

    return out_dict_LTE

def calc_one_LTE(
        T_kin = 25., 
        N_CO = 1e18, 
        X_CICO = 0.2, 
        n_H2 = 1e2, 
        d_V = 5., 
        T_CMB = 2.72548, 
        overwrite = False, 
        verbose = True, 
        backup = True, 
        write_lock = None, 
    ):
    # 
    # run calculation (using a sqliteutil module in ../utils/ for cache)
    output_dict = {}
    db_file = os.path.join(LTE_calc_path, 'calc_LTE.db')
    calc_func = calc_one_LTE_one_species
    with sqliteutil.SqliteUtil(db_file, 
                               calc_func, 
                               verbose = verbose, 
                               backup = backup, 
                               write_lock = write_lock, 
                              ) as calc_util:
        output_dict['CO'] = calc_util.get_CO(T_kin, N_CO, n_H2, d_V, 
                                             overwrite = overwrite, verbose = verbose, 
                                             LTE_calc_path = LTE_calc_path, 
                                             moldata_object = CO)
        output_dict['CI'] = calc_util.get_CI_by_X_CICO(T_kin, N_CO, X_CICO, n_H2, d_V, 
                                             overwrite = overwrite, verbose = verbose, 
                                             LTE_calc_path = LTE_calc_path, 
                                             moldata_object = CI)
    # 
    # return
    return output_dict

# test
out_dict_test = None
if True:
    out_dict_test = calc_one_LTE(
            T_kin = list_of_T_kin[0], 
            N_CO = list_of_N_CO[0], 
            X_CICO = list_of_X_CICO[0], 
            n_H2 = list_of_n_H2[0], 
            d_V = line_width, 
            overwrite = True, 
        )
#Table(out_dict_test['CI'])
Table(out_dict_test['CO'])[0:16]


# In[272]:


# Define function to compute non-LTE solutions for one (T_kin, N_CO/line_width, n_H2, X_CICO)

import sqliteutil
import importlib
importlib.reload(sqliteutil)

def exec_one_RADEX_command(
        working_dir, 
        radex_bin_path, 
        calc_inp_file, 
        command_str, 
    ):
    ntry = 10
    failed = False
    while ntry > 0:
        failed = False
        #subprocess.check_output(command_str, shell=True, executable='/bin/bash')
        #subprocess.call does not work under linux
        with subprocess.Popen([radex_bin_path], stdout=subprocess.PIPE, 
                              stdin=subprocess.PIPE, stderr=subprocess.PIPE, 
                              cwd=working_dir) as proc:
            with open(calc_inp_file, 'r') as fp:
                for line in fp:
                    proc.stdin.write(line.encode()) # line already has '\n' ending
            try:
                proc_out, proc_err = proc.communicate(timeout=15)
                failed = (proc.returncode != 0)
                #print('proc.returncode', proc.returncode)
                #print('proc_out', proc_out)
                #print('proc_err', proc_err)
            except subprocess.TimeoutExpired as err:
                failed = True
            if not failed:
                break
        ntry -= 1
    if failed:
        raise Exception('Error! Failed to run the RADEX calculation: ' + command_str + '\n' + 
                        '... stderr: ' + '\n' + proc_err.decode() + '\n' + 
                        '... stdout: ' + '\n' + proc_out.decode() + '\n')
    return (not failed)

def calc_one_RADEX_one_species(
        working_dir, species_name, T_CMB, T_kin, N_X, n_H2, d_V, overwrite, verbose, # must have these args
        radex_bin_path = '', # then kwargs
        radex_data_path = '', 
        radex_calc_path = '', 
        radex_data_file = '', 
        script_name = '',
    ):
    # 
    # prepare radex data files: co.dat catom.dat
    if radex_calc_path != '':
        working_dir = os.path.join(radex_calc_path, working_dir)
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir, exist_ok=True)
    calc_dat_file = os.path.join(working_dir, radex_data_file)
    if not os.path.isfile(calc_dat_file):
        shutil.copy2(os.path.join(radex_data_path, radex_data_file), 
                     calc_dat_file)
    if not os.path.isfile(calc_dat_file): # double check sometimes needed
        shutil.copy2(os.path.join(radex_data_path, radex_data_file), 
                     calc_dat_file)
    else:
        if os.path.getsize(calc_dat_file) == 0:
            shutil.copy2(os.path.join(radex_data_path, radex_data_file), 
                         calc_dat_file)
    # 
    # prepare radex input files: *.inp
    if script_name == '':
        script_name = species_name.lower()
    calc_inp_file = os.path.join(working_dir, script_name + '.inp')
    calc_out_file = os.path.join(working_dir, script_name + '.out')
    if os.path.isfile(calc_inp_file) and overwrite:
        shutil.move(calc_inp_file, calc_inp_file + '.bak')
    if os.path.isfile(calc_out_file) and overwrite:
        shutil.move(calc_out_file, calc_out_file + '.bak')
    if not os.path.isfile(calc_inp_file) or True:
        with open(calc_inp_file, 'w') as fp:
            fp.write('./' + radex_data_file + '\n')
            fp.write(script_name + '.out\n')
            fp.write('100 1900\n') # GHz range
            fp.write('%s\n'%(T_kin)) # 
            fp.write('2\n') # collision partner number
            fp.write('o-H2\n')
            fp.write('%.10e\n'%(n_H2 * 0.75)) # 2371708.245
            fp.write('p-H2\n')
            fp.write('%.10e\n'%(n_H2 * 0.25)) # 790569.415
            fp.write('%s\n'%(T_CMB))
            fp.write('%.10e\n'%(N_X))
            fp.write('%s\n'%(d_V))
            fp.write('0\n\n') # do not do another calc
    # run radex to produce the output file if it does not exist
    if not os.path.isfile(calc_out_file):
        command_str = 'cd "%s"; %s < %s 2>&1 > %s'%(working_dir, radex_bin_path, 
                                               script_name + '.inp', script_name + '.log')
        debug = True
        if verbose or debug:
            print('command_str: ' + command_str)
        # 
        exec_one_RADEX_command(
            working_dir, 
            radex_bin_path, 
            calc_inp_file, 
            command_str, 
        )
    # 
    # prepare output dict
    calc_out_dict = OrderedDict([ 
        ('J_UP',[]), 
        ('J_LOW',[]), 
        ('E_UP',[]), 
        ('FREQ',[]), 
        ('WAVE',[]), 
        ('T_ex',[]), 
        ('tau_0',[]), 
        ('T_RJ',[]), 
        ('Pop_u',[]), 
        ('Pop_l',[]), 
        ('Flux_Kkms',[]), 
        ('Flux_ergcm2s',[]), 
    ])
    # 
    # read radex output file content
    if verbose:
        print('reading ' + calc_out_file)
    with open(calc_out_file, 'r') as fp:
        for line_str in fp:
            if line_str.startswith('#') or line_str.startswith('*'):
                continue
            if line_str.startswith('J_UP') or re.match(r'^\s+(LINE|\(K\)).*', line_str):
                continue
            line_split = line_str.split()
            line_split = [t for t in line_split if t != '--']
            if len(line_split) >= 12:
                for icol, key in enumerate(calc_out_dict.keys()):
                    calc_out_dict[key].append(float(line_split[icol]))
    # cleanup
    shutil.rmtree(working_dir)
    # return
    return calc_out_dict

def calc_one_RADEX(
        T_kin = 25., 
        N_CO = 1e18, 
        X_CICO = 0.2, 
        n_H2 = 1e2, 
        d_V = 5., 
        T_CMB = 2.72548, 
        overwrite = False, 
        verbose = True, 
        backup = True, 
        write_lock = None, 
    ):
    # 
    # run calculation (using a sqliteutil module in ../utils/ for cache)
    output_dict = {}
    db_file = os.path.join(radex_calc_path, 'calc_radex.db')
    calc_func = calc_one_RADEX_one_species
    with sqliteutil.SqliteUtil(db_file, 
                               calc_func, 
                               verbose = verbose, 
                               backup = backup, 
                               write_lock = write_lock, 
                              ) as calc_util:
        output_dict['CO'] = calc_util.get_CO(T_kin, N_CO, n_H2, d_V, 
                                             overwrite = overwrite, verbose = verbose, 
                                             radex_bin_path = radex_bin_path, 
                                             radex_data_path = radex_data_path, 
                                             radex_calc_path = radex_calc_path, 
                                             radex_data_file = 'co.dat')
        output_dict['CI'] = calc_util.get_CI_by_X_CICO(T_kin, N_CO, X_CICO, n_H2, d_V, 
                                             overwrite = overwrite, verbose = verbose, 
                                             radex_bin_path = radex_bin_path, 
                                             radex_data_path = radex_data_path, 
                                             radex_calc_path = radex_calc_path, 
                                             radex_data_file = 'catom.dat')
    # 
    # return
    return output_dict

# test
out_dict_test = None
if True:
    out_dict_test = calc_one_RADEX(
            T_kin = list_of_T_kin[0], 
            N_CO = list_of_N_CO[0], 
            X_CICO = list_of_X_CICO[0], 
            n_H2 = list_of_n_H2[0], 
            d_V = line_width, 
            overwrite = False, 
        )
Table(out_dict_test['CO'])


# In[273]:


# Define function to compute non-LTE and LTE solutions for one (T_kin, N_CO/line_width, n_H2, X_CICO) input set

def run_for_one_input_set(
        var_dict, 
        overwrite = False, 
        verbose = True, 
        backup = True, 
        write_lock = None, 
    ):
    # 
    global radex_calc_path
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
    
    # do the LTE calculation
    out_dict_LTE = calc_one_LTE(
        T_kin = T_kin.value, 
        N_CO = N_CO.value, 
        X_CICO = N_CI.value/N_CO.value, 
        n_H2 = n_H2.value, 
        d_V = d_V.value, 
        overwrite = overwrite, 
        verbose = verbose, 
        backup = backup,  
        write_lock = write_lock, 
    )
    out_table_CI_LTE = out_dict_LTE['CI']
    out_table_CO_LTE = out_dict_LTE['CO']
    
    # do the RADEX NonLTE calculation
    out_dict_RADEX = calc_one_RADEX(
        T_kin = T_kin.value, 
        N_CO = N_CO.value, 
        X_CICO = N_CI.value/N_CO.value, 
        n_H2 = n_H2.value, 
        d_V = d_V.value, 
        overwrite = overwrite, 
        verbose = verbose, 
        backup = backup, 
        write_lock = write_lock, 
    )
    out_table_CI_NonLTE = out_dict_RADEX['CI']
    out_table_CO_NonLTE = out_dict_RADEX['CO']
    
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
                'collision_partner_densities': {'ortho-H2': 0.75 * list_of_n_H2[0] * u.cm**(-3), 
                                                'para-H2': 0.25 * list_of_n_H2[0] * u.cm**(-3)}, 
            }
        )
res_dict


# In[274]:


# Define function to write a table to disk

def write_table_to_disk(out_table, out_table_name):
    
    global radex_calc_path
    
    if not os.path.isdir(radex_calc_path):
        os.makedirs(radex_calc_path)
    out_table_file = os.path.join(radex_calc_path, out_table_name)

    if os.path.isfile(out_table_file):
        shutil.move(out_table_file, out_table_file+'.backup')
    
    if isinstance(out_table, (dict, OrderedDict)):
        out_table = Table(out_table)
    
    out_table.write(out_table_file + '.csv', format='csv', overwrite=True)
    #out_table.write(out_table_file + '.dat', format='ascii.fixed_width', delimiter='  ', 
    #                bookend=True, overwrite=True)
    print('Output to "%s"'%(out_table_file + '.csv'))
    #print('Output to "%s"'%(out_table_file + '.dat'))


# In[275]:


# Define callback

def callback(x):
    
    pass
    

# Define lookup key format function

def format_lookup_key(X_CICO, d_V):
    if X_CICO < 0.1:
        X_CICO_str = '{:.2f}'.format(X_CICO)
    else:
        X_CICO_str = '{:.1f}'.format(X_CICO)
    if d_V < 0.1:
        d_V_str = '{:.2f}'.format(d_V)
    else:
        d_V_str = '{:.1f}'.format(d_V)
    
    return f'XCICO_{X_CICO_str}_dv_{d_V_str}kms'


# In[276]:


# Run the calculation loop

#from tqdm.notebook import tqdm
from IPython.display import clear_output, display
import multiprocessing as mp
import gc

def run_for_all_input_sets():
    
    global CO
    global CI
    global list_of_X_CICO
    global list_of_N_CO
    global list_of_n_H2
    global list_of_T_kin
    global CPU_cores
    
    mPool = mp.Pool(CPU_cores)
    mManager = mp.Manager()
    mLock = mManager.Lock()

    idx_all = 0
    n_all = len(list_of_X_CICO) * len(list_of_d_V) * len(list_of_N_CO) * len(list_of_n_H2) * len(list_of_T_kin)
    
    # Loop
    for idx_X_CICO, X_CICO in enumerate(list_of_X_CICO):
        
        # Loop
        for idx_d_V, d_V in enumerate(list_of_d_V):
            
            lookup_key = format_lookup_key(X_CICO, d_V)
            out_table_name = 'out_table_RADEX_R_CI10CO21_sim_' + lookup_key
            out_table_file = os.path.join(radex_calc_path, out_table_name + '.csv')
            
            if os.path.isfile(out_table_file):
                idx_all += len(list_of_N_CO) * len(list_of_n_H2) * len(list_of_T_kin)
                print('Found ' + out_table_file + ', skip!')
                print('')
                continue
            
            backup_on_first_run = True
            
            list_of_var_dict = []
            list_of_input_set = []
            var_dict = {'CO': CO, 'CI': CI}
            out_dict = None
            
            for idx_N_CO, N_CO in enumerate(list_of_N_CO):
                for idx_n_H2, n_H2 in enumerate(list_of_n_H2):
                    for idx_T_kin, T_kin in enumerate(list_of_T_kin):
                        var_dict['N_CO'] = N_CO * u.cm**(-2)
                        var_dict['N_CI'] = X_CICO * N_CO * u.cm**(-2) # CI/CO intrinsic abundance ratio
                        var_dict['X_CICO'] = X_CICO
                        var_dict['d_V'] = d_V * u.km/u.s
                        var_dict['T_kin'] = T_kin * u.K
                        var_dict['n_H2'] = n_H2 * u.cm**(-3)
                        var_dict['collision_partner_densities'] = {'ortho-H2': 0.75 * n_H2 * u.cm**(-3), 
                                                                   'para-H2': 0.25 * n_H2 * u.cm**(-3)}
                        list_of_var_dict.append(copy.copy(var_dict))
                        
                        N_CO_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', '{:.2e}'.format(N_CO))
                        n_H2_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', '{:.2e}'.format(n_H2))
                        input_set_str =                         '{:>4.1f} {:>5.1f} {:>8s} {:>7s} {:>5.1f} | '                         '{:02d}/{:02d} {:02d}/{:02d} {:02d}/{:02d} {:02d}/{:02d} {:02d}/{:02d} | {:06d}/{:06d}'                         .format(
                            X_CICO, d_V, N_CO_str, n_H2_str, T_kin, 
                            idx_X_CICO + 1, len(list_of_X_CICO), 
                            idx_d_V + 1, len(list_of_d_V), 
                            idx_N_CO + 1, len(list_of_N_CO), 
                            idx_n_H2 + 1, len(list_of_n_H2), 
                            idx_T_kin + 1, len(list_of_T_kin), 
                            idx_all + 1, n_all, 
                        )
                        idx_all += 1
                        list_of_input_set.append(input_set_str)

            # run the processes
            list_of_res_async = []
            for i in range(len(list_of_var_dict)):
                overwrite = False
                verbose = False

                clear_output(wait=True)
                print('Adding process: ' + list_of_input_set[i])

                res = mPool.apply_async(run_for_one_input_set, 
                                        (list_of_var_dict[i], overwrite, verbose, backup_on_first_run, mLock), 
                                       )

                backup_on_first_run = False
                list_of_res_async.append(res)

            # retrieve the results in order
            list_of_res_dict = []
            for i in range(len(list_of_var_dict)):

                clear_output(wait=True)
                print('Retrieving result: ' + list_of_input_set[i])

                res_dict = list_of_res_async[i].get()
                list_of_res_dict.append(res_dict)
            
            # concatenate table
            out_dict = OrderedDict()
            for key in list(list_of_res_dict[0].keys()):
                if isinstance(list_of_res_dict[0][key], u.Quantity):
                    out_dict[key] = [res_dict[key].value for res_dict in list_of_res_dict]
                else:
                    out_dict[key] = [res_dict[key] for res_dict in list_of_res_dict]
            
            # write to disk
            write_table_to_disk(out_dict, out_table_name)

            #break
            
            del out_dict
            del res_dict
            del list_of_res_dict
            del list_of_var_dict
            del list_of_input_set
            gc.collect()
        
        #break

    #out_table


# In[277]:


# Main Program Entry

if __name__ == '__main__':
    
    run_for_all_input_sets()


# In[ ]:




