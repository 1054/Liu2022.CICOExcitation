#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, re, shutil
import numpy as np
import sqlite3 as sqlite
import astropy.units as u
from collections import OrderedDict
from typing import Callable

class SqliteUtil():
    
    """docstring for SqliteUtil"""
    
    def __init__(self, 
            db_file:str, 
            calc_func:Callable, 
            verbose:bool = True, 
            backup:bool = True, 
            write_lock = None, 
            column_formats:dict = None, 
        ):
        # 
        if db_file.find(os.sep)>=0:
            if not os.path.isdir(os.path.dirname(db_file)):
                os.makedirs(os.path.dirname(db_file))
        # 
        self.db_file = db_file
        self.db_conn = None
        self.db_cursor = None
        self.calc_func = calc_func
        self.write_lock = write_lock
        self.column_formats = column_formats
        if verbose:
            print('db_file: ' + db_file)
        if backup:
            if os.path.isfile(db_file):
                shutil.copy2(db_file, db_file + '.backup')
                if verbose:
                    print('Backing-up ' + db_file + ' as ' + db_file + '.backup')
        
        self.connect()
    
    def connect(self):
        self.db_conn = sqlite.connect(self.db_file)
        self.db_cursor = self.db_conn.cursor()
    
    def disconnect(self):
        if self.db_conn is not None:
            #self.db_cursor.close()
            self.db_conn.close()
            self.db_conn = None
            self.db_cursor = None
    
    def __del__(self):
        self.disconnect()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exct_type, exce_value, traceback):
        self.disconnect()
    
    def create_table(self, species_name:str, exist_ok:bool = True, verbose:bool = True):
        # check if co table exists
        sqlcmd = "SELECT count(name) FROM sqlite_master WHERE type='table' AND name='%s'"%(species_name)
        
        # check the write lock
        if self.write_lock is not None:
            self.write_lock.acquire()
        
        # execute and fetchone
        if verbose:
            print('sqlcmd: ' + sqlcmd)
        self.db_cursor.execute(sqlcmd)
        queried_rows = self.db_cursor.fetchone()
        
        # release write lock
        if self.write_lock is not None:
            self.write_lock.release()
        
        if queried_rows[0] == 0:
            sqlcmd = 'CREATE TABLE ' + species_name + ' ('
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
            
            # check the write lock
            if self.write_lock is not None:
                self.write_lock.acquire()
            
            # execute and commit
            if verbose:
                print('sqlcmd: ' + sqlcmd)
            self.db_cursor.execute(sqlcmd)
            self.db_conn.commit()
            
            # release write lock
            if self.write_lock is not None:
                self.write_lock.release()
    
    def add_row_to_table(self, species_name:str, T_CMB:float, T_kin:float, N_X:float, n_H2:float, d_V:float, calc_out_dict:dict, verbose:bool = True):
        # check if the table exists
        calc_name = self.format_calc_name(species_name, T_CMB, T_kin, N_X, n_H2, d_V)
        colnames = list(calc_out_dict.keys())
        nrows = len(calc_out_dict[colnames[0]])
        sqlcmds = []
        if isinstance(T_CMB, u.Quantity):
            T_CMB = T_CMB.to(u.K).value
        if isinstance(T_kin, u.Quantity):
            T_kin = T_kin.to(u.K).value
        if isinstance(N_X, u.Quantity):
            N_X = N_X.to(u.cm**(-2)).value
        if isinstance(n_H2, u.Quantity):
            n_H2 = n_H2.to(u.cm**(-3)).value
        if isinstance(d_V, u.Quantity):
            d_V = d_V.to(u.km/u.s).value
        for i in range(nrows):
            Ju_str = re.sub(r'[^0-9a-zA-Z.]+', r'_', "%g"%(calc_out_dict['J_UP'][i]))
            Jl_str = re.sub(r'[^0-9a-zA-Z.]+', r'_', "%g"%(calc_out_dict['J_LOW'][i]))
            calc_name_Ju_Jl = calc_name + '_Ju_' + Ju_str + '_Jl_' + Jl_str
            sqlcmd = 'INSERT OR REPLACE INTO ' + species_name + ' VALUES ('
            sqlcmd += "'" + calc_name_Ju_Jl + "', "
            sqlcmd += "'" + calc_name + "', "
            sqlcmd += "'" + species_name + "', "
            sqlcmd += str(T_CMB) + ", "
            sqlcmd += str(T_kin) + ", "
            sqlcmd += str(N_X) + ", "
            sqlcmd += str(n_H2) + ", "
            sqlcmd += str(d_V) + ", "
            for icol, colname in enumerate(colnames):
                val = calc_out_dict[colname][i] # float value
                if isinstance(val, u.Quantity):
                    val = val.value
                sqlcmd += str(val) # float value
                if icol < len(colnames)-1:
                     sqlcmd += ", "
            sqlcmd += ');'
            sqlcmds.append(sqlcmd)
        
        # check the write lock
        # note that according to https://www.sqlite.org/faq.html :
        # "Multiple processes can be doing a SELECT at the same time. But only one process can be 
        # making changes to the database at any moment in time, however."
        if self.write_lock is not None:
            self.write_lock.acquire()
        
        for i in range(nrows):
            sqlcmd = sqlcmds[i]
            if verbose: #<TODO># always print INSERT command
                print('sqlcmd: ' + sqlcmd)
            self.db_cursor.execute(sqlcmd)
        self.db_conn.commit()
            
        # release write lock
        if self.write_lock is not None:
            self.write_lock.release()
    
    def format_calc_name(self, species_name:str, T_CMB:float, T_kin:float, N_X:float, n_H2:float, d_V:float):
        T_CMB_fmt = '{:.2f}'
        T_kin_fmt = '{:.1f}'
        N_X_fmt = '{:.2e}'
        n_H2_fmt = '{:.2e}'
        d_V_fmt = '{:.1f}'
        if isinstance(self.column_formats, (dict, OrderedDict)) and len(self.column_formats) > 0:
            if 'T_CMB' in self.column_formats:
                T_CMB_fmt = self.column_formats['T_CMB']
            if 'T_kin' in self.column_formats:
                T_kin_fmt = self.column_formats['T_kin']
            if 'N_X' in self.column_formats:
                N_X_fmt = self.column_formats['N_X']
            if 'n_H2' in self.column_formats:
                n_H2_fmt = self.column_formats['n_H2']
            if 'd_V' in self.column_formats:
                d_V_fmt = self.column_formats['d_V']
        T_CMB_str = T_CMB_fmt.format(T_CMB)
        T_kin_str = T_kin_fmt.format(T_kin)
        N_X_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', N_X_fmt.format(N_X))
        n_H2_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', n_H2_fmt.format(n_H2))
        d_V_str = d_V_fmt.format(d_V)
        calc_name = 'calc_T_CMB_%s_T_kin_%s_N_%s_%s_n_H2_%s_d_V_%s'%(T_CMB_str, T_kin_str, species_name, N_X_str, n_H2_str, d_V_str)
        return calc_name
    
    def format_working_dir(self, species_name:str, T_CMB:float, T_kin:float, N_X:float, n_H2:float, d_V:float):
        # T_kin_str = '%.0f'%(T_kin)
        # N_X_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', '%.2e'%(N_X))
        # n_H2_str = re.sub(r'(.*)(e)(\+0|\+)([0-9]+.*)', r'\1\2\4', '%.1e'%(n_H2))
        # d_V_str = '%.1f'%(d_V)
        # working_dir = 'calc_T_kin_%s_N_%s_%s_n_H2_%s_d_V_%s'%(T_kin_str, species_name, N_X_str, n_H2_str, d_V_str)
        # return working_dir
        return self.format_calc_name(species_name, T_CMB, T_kin, N_X, n_H2, d_V)
    
    def calc_one_species(self, species_name:str, T_CMB:float, T_kin:float, N_X:float, n_H2:float, d_V:float, 
            overwrite:bool = False, verbose:bool = False, **kwargs):
        # 
        self.create_table(species_name, exist_ok=True, verbose=verbose)
        
        if verbose:
            print('calc_one_species')
        
        # get working_dir
        working_dir = self.format_working_dir(species_name, T_CMB, T_kin, N_X, n_H2, d_V)
        
        # for historical reasons, for 'CI' we use the CO working_dir, and 'ci_X_CICO_*' as the script_name. 
        # if species_name == 'CI' and 'X_CICO' in kwargs: 
        #     X_CICO = kwargs['X_CICO']
        #     del kwargs['X_CICO']
        #     X_CICO_str = '%.2f'%(X_CICO)
        #     N_CO = N_X / X_CICO
        #     working_dir = self.format_working_dir('CO', T_CMB, T_kin, N_CO, n_H2, d_V)
        #     kwargs.update({'script_name': 'ci_X_CICO_%s'%(X_CICO_str)})
        
        if verbose:
            print('working_dir: ' + working_dir)
        
        # do the computation and get result table
        calc_out_dict = self.calc_func(working_dir, species_name, T_CMB, T_kin, N_X, n_H2, d_V, overwrite, verbose, **kwargs)
        
        # store into database
        self.add_row_to_table(species_name, T_CMB, T_kin, N_X, n_H2, d_V, calc_out_dict, verbose)
        
        # return 
        return calc_out_dict
    
    def query_one_species(self, species_name:str, T_CMB:float, T_kin:float, N_X:float, n_H2:float, d_V:float, 
            overwrite:bool = False, verbose:bool = False, **kwargs
        ):
        """Return a dictionary in which each item is a list.
        """
        #self.create_table(species_name, exist_ok=True, verbose=verbose)
        
        calc_name = self.format_calc_name(species_name, T_CMB, T_kin, N_X, n_H2, d_V)
        
        sqlcmd = "SELECT * FROM %s WHERE calc_name='%s'"%(species_name, calc_name)
        if verbose:
            print('sqlcmd: ' + sqlcmd)
        try:
            self.db_cursor.execute(sqlcmd)
            queried_rows = self.db_cursor.fetchall()
        except sqlite.OperationalError as err:
            queried_rows = None
        
        if queried_rows is None or queried_rows == []:
            calc_out = self.calc_one_species(species_name, T_CMB, T_kin, N_X, n_H2, d_V, overwrite, verbose, **kwargs)
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
                #print('row: ' + str(row))
                queried_dict = dict(list(zip(columns, row)))
                for key in calc_out:
                    calc_out[key].append(queried_dict[key])
        
        return calc_out
    
    def get_CO(self, T_kin:float, N_CO:float, n_H2:float, d_V:float, overwrite:bool = False, verbose:bool = True, **kwargs):
        species_name = 'CO'
        N_X = N_CO
        T_CMB = 2.72548
        return self.query_one_species(species_name, T_CMB, T_kin, N_X, n_H2, d_V, overwrite, verbose, **kwargs)
    
    def get_CI_by_X_CICO(self, T_kin:float, N_CO:float, X_CICO:float, n_H2:float, d_V:float, overwrite:bool = False, verbose:bool = True, **kwargs):
        species_name = 'CI'
        # kwargs.update({'X_CICO':X_CICO}) # for historical reasons, we need this
        N_X = N_CO * X_CICO
        T_CMB = 2.72548
        return self.query_one_species(species_name, T_CMB, T_kin, N_X, n_H2, d_V, overwrite, verbose, **kwargs)
    





