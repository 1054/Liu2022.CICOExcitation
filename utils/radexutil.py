#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, re, shutil
import numpy as np
import sqlite3 as sqlite
from astropy.table import Table
from collections import OrderedDict

#sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
#from sqliteutil import SqliteUtil

class RadexUtil():
    
    """docstring for RadexUtil"""
    
    def __init__(self, 
            radex_bin_path, 
            radex_data_path, 
            radex_calc_path = None, 
            radex_database = None, 
        ):
        
        self.radex_bin_path = radex_bin_path
        assert os.path.isfile(self.radex_bin_path)
        
        self.radex_data_path = radex_data_path
        assert os.path.isdir(self.radex_data_path)
        
        self.radex_data_dict = {}
        self.radex_data_dict['CO'] = os.path.join(self.radex_data_path, 'co.dat')
        self.radex_data_dict['CI'] = os.path.join(self.radex_data_path, 'catom.dat')
        #assert os.path.isfile(self.radex_data_dict['CO'])
        #assert os.path.isfile(self.radex_data_dict['CI'])
        
        if radex_calc_path is None:
            self.radex_calc_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'tmp' + os.sep + 'calc_radex'
        else:
            self.radex_calc_path = radex_calc_path
        if radex_database is None:
            self.radex_database = 'calc_radex.db'
        else:
            self.radex_database = radex_database
        # 
        self.db_file = os.path.join(self.radex_calc_path, self.radex_database)
        self.db_conn = sqlite.connect(self.db_file)
        self.db_cursor = self.db_conn.cursor()
        # 
        self.T_CMB = 2.725
    
    def __del__(self):
        #self.db_cursor.close()
        self.db_conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exct_type, exce_value, traceback):
        #self.db_cursor.close()
        self.db_conn.close()
    
    def create_table(self, name:str, exist_ok:bool = True):
        # check if co table exists
        sqlcmd = "SELECT count(name) FROM sqlite_master WHERE type='table' AND name='%s'"%(name)
        print('sqlcmd: ' + sqlcmd)
        self.db_cursor.execute(sqlcmd)
        
        if self.db_cursor.fetchone()[0] == 0:
            sqlcmd = 'CREATE TABLE ' + name + ' ('
            sqlcmd += 'calc_name_Ju_Jl text PRIMARY KEY,'
            sqlcmd += 'calc_name text NOT NULL,'
            sqlcmd += 'species text NOT NULL,'
            sqlcmd += 'T_CMB REAL NOT NULL,'
            sqlcmd += 'T_kin REAL NOT NULL,'
            sqlcmd += 'N_X REAL NOT NULL,'
            sqlcmd += 'n_H2 REAL NOT NULL,'
            sqlcmd += 'd_V REAL NOT NULL,'
            sqlcmd += 'J_UP REAL NOT NULL,'
            sqlcmd += 'J_LOW REAL NOT NULL,'
            sqlcmd += 'E_UP REAL NOT NULL,'
            sqlcmd += 'FREQ REAL NOT NULL,'
            sqlcmd += 'WAVE REAL NOT NULL,'
            sqlcmd += 'T_ex REAL NOT NULL,'
            sqlcmd += 'tau_0 REAL NOT NULL,'
            sqlcmd += 'T_RJ REAL NOT NULL,'
            sqlcmd += 'Pop_u REAL NOT NULL,'
            sqlcmd += 'Pop_l REAL NOT NULL,'
            sqlcmd += 'Flux_Kkms REAL NOT NULL,'
            sqlcmd += 'Flux_ergcm2s REAL NOT NULL' # no comma at the end
            sqlcmd += ');'
            print('sqlcmd: ' + sqlcmd)
            self.db_cursor.execute(sqlcmd)
            self.db_conn.commit()
    
    def format_species_Ju(self, name:str, Ju:float):
        if name in ['CO', 'CI']:
            Ju_str = re.sub(r'[^0-9a-zA-Z.]+', r'_', str(int(Ju)).strip())
        else:
            Ju_str = re.sub(r'[^0-9a-zA-Z.]+', r'_', str(Ju).strip())
        
        return Ju_str
    
    def add_row_to_table(self, name:str, T_kin:float, N_X:float, n_H2:float, d_V:float, calc_out_table:Table):
        # check if co table exists
        calc_name = self.format_calc_name(name, T_kin, N_X, n_H2, d_V)
        for i in range(len(calc_out_table)):
            Ju_str = self.format_species_Ju(name, calc_out_table['J_UP'][i])
            Jl_str = self.format_species_Ju(name, calc_out_table['J_LOW'][i])
            calc_name_Ju_Jl = calc_name + '_Ju_' + Ju_str + '_Jl_' + Jl_str
            sqlcmd = 'INSERT OR REPLACE INTO ' + name + ' VALUES ('
            sqlcmd += "'" + calc_name_Ju_Jl + "', "
            sqlcmd += "'" + calc_name + "', "
            sqlcmd += "'" + name + "', "
            sqlcmd += str(self.T_CMB) + ", "
            sqlcmd += str(T_kin) + ", "
            sqlcmd += str(N_X) + ", "
            sqlcmd += str(n_H2) + ", "
            sqlcmd += str(d_V) + ", "
            for icol, colname in enumerate(list(calc_out_table.colnames)):
                sqlcmd += str(calc_out_table[colname][i]) # float value
                if icol < len(calc_out_table.colnames)-1:
                     sqlcmd += ", "
            sqlcmd += ');'
            print('sqlcmd: ' + sqlcmd)
            self.db_cursor.execute(sqlcmd)
        self.db_conn.commit()
    
    def format_calc_name(self, name:str, T_kin:float, N_X:float, n_H2:float, d_V:float):
        T_kin_str = '%.0f'%(T_kin)
        N_X_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', '%.2e'%(N_X))
        n_H2_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', '%.1e'%(n_H2))
        d_V_str = '%.1f'%(d_V)
        calc_name = 'calc_T_kin_%s_N_%s_%s_n_H2_%s_d_V_%s'%(T_kin_str, name, N_X_str, n_H2_str, d_V_str)
        return calc_name
    
    def calc_one_species(self, name:str, fname:str, T_kin:float, N_X:float, n_H2:float, d_V:float, overwrite:bool = False, return_dict:bool=False):
        
        T_CMB = self.T_CMB
        
        assert name in self.radex_data_dict
        
        radex_calc_path = self.radex_calc_path
        radex_bin_path = self.radex_bin_path
        radex_data_file = self.radex_data_dict[name]
        print('radex_calc_path: ' + radex_calc_path)
        print('radex_bin_path: ' + radex_bin_path)
        print('radex_data_file: ' + radex_data_file)
        
        calc_name = self.format_calc_name(name, T_kin, N_X, n_H2, d_V)
        print('calc_name: ' + calc_name)
        
        # prepare working directory
        working_dir = os.path.join(radex_calc_path, calc_name)
        
        if name == 'CI': # for 'CI' for historical reason we use CO working_dir
            X_CICO = float(re.sub(r'^ci_X_CICO_([0-9.]+)$', r'\1', fname))
            N_CO = N_X / X_CICO
            calc_name_CO = self.format_calc_name('CO', T_kin, N_CO, n_H2, d_V)
            working_dir = os.path.join(radex_calc_path, calc_name_CO)
        
        if not os.path.isdir(working_dir):
            os.makedirs(working_dir)
        
        # prepare radex data file
        calc_data_name = os.path.basename(radex_data_file)
        calc_data_file = os.path.join(radex_calc_path, calc_data_name)
        if not os.path.isfile(calc_data_file):
            shutil.copy2(radex_data_file, calc_data_file)
        
        # prepare radex input files
        calc_inp_file = os.path.join(working_dir, fname + '.inp')
        calc_out_file = os.path.join(working_dir, fname + '.out')
        if os.path.isfile(calc_inp_file) and overwrite:
            shutil.move(calc_inp_file, calc_inp_file + '.bak')
        if os.path.isfile(calc_out_file) and overwrite:
            shutil.move(calc_out_file, calc_out_file + '.bak')
        if not os.path.isfile(calc_inp_file):
            with open(calc_inp_file, 'w') as fp:
                fp.write('../' + calc_data_name + '\n')
                fp.write(fname + '.out\n')
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
        
        #with open(os.path.join(working_dir, fname + '.inp'), 'r') as fp:
        #    calc_inp = ''.join(fp.readlines())
        
        # run radex to produce the output file if it does not exist
        if not os.path.isfile(os.path.join(working_dir, fname + '.out')):
            command_str = 'cd "%s"; %s < %s > %s'%(working_dir, radex_bin_path, fname + '.inp', fname + '.log')
            print('command_str:' + command_str)
            subprocess.call(command_str, shell=True, executable='/bin/bash')
        
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
        ]) # must have the same columns and order as in self.create_table
        
        # read radex output file content
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
        
        calc_out_table = Table(calc_out_dict)
        
        # store into database
        self.add_row_to_table(name, T_kin, N_X, n_H2, d_V, calc_out_table)
        
        # clean up
        #shutil.rmtree(working_dir)
        
        # return 
        if return_dict:
            return calc_out_dict
        return calc_out_table
    
    def query_one_species(self, name:str, fname:str, T_kin:float, N_X:float, n_H2:float, d_V:float, overwrite:bool = False):
        """Return a dictionary in which each item is a list.
        """
        self.create_table(name)
        
        calc_name = self.format_calc_name(name, T_kin, N_X, n_H2, d_V)
        
        sqlcmd = "SELECT * FROM %s WHERE calc_name='%s'"%(name, calc_name)
        print('sqlcmd: ' + sqlcmd)
        self.db_cursor.execute(sqlcmd)
        
        queried_rows = self.db_cursor.fetchall()
        
        if queried_rows is None or queried_rows == []:
            calc_out = self.calc_one_species(name, fname, T_kin, N_X, n_H2, d_V, overwrite = overwrite, return_dict = True)
        else:
            calc_out = OrderedDict([ 
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
            ]) # must have the same columns and order as in self.create_table
            columns = [desc[0] for desc in self.db_cursor.description]
            for row in queried_rows:
                print('row: ' + str(row))
                queried_dict = dict(list(zip(columns, row)))
                for key in calc_out:
                    calc_out[key].append(queried_dict[key])
        
        return calc_out
    
    def get_CO(self, T_kin:float, N_CO:float, n_H2:float, d_V:float, overwrite:bool = False):
        name = 'CO'
        fname = 'co'
        N_X = N_CO
        return self.query_one_species(name, fname, T_kin, N_X, n_H2, d_V, overwrite)
    
    def get_CI_by_X_CICO(self, T_kin:float, N_CO:float, X_CICO:float, n_H2:float, d_V:float, overwrite:bool = False):
        X_CICO_str = '%.2f'%(X_CICO)
        name = 'CI'
        fname = 'ci_X_CICO_%s'%(X_CICO_str)
        N_X = N_CO * X_CICO
        return self.query_one_species(name, fname, T_kin, N_X, n_H2, d_V, overwrite)
    





