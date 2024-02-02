#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, re, copy
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.modeling.models import BlackBody
# from astropy.modeling.blackbody import blackbody_nu
from astropy.table import Table
from collections import OrderedDict, namedtuple
from dataclasses import dataclass # Python >= 3.7
from pprint import pprint, PrettyPrinter
from scipy.interpolate import UnivariateSpline, interp1d
import emcee
import corner
import warnings
warnings.filterwarnings('ignore')

# Level = namedtuple('Level', ['energy', 'weight', 'J', 'column_density'])
# Transition = namedtuple('Transition', ['J_u', 'J_l', 'Einstein_A', 'Freq', 'E_u', 'tau_0', 'T_ex', 'P_lu', 'P_ul'])
# Collision = namedtuple('Collision', ['J_u', 'J_l', 'rate_spl', 'rate'])
@dataclass
class Level():
    energy: u.Quantity
    weight: float
    J: float
    column_density: u.Quantity
    pop_in_rate_A: dict
    pop_in_rate_B: dict
    pop_in_rate_C: dict
    pop_out_rate_A: dict
    pop_out_rate_B: dict
    pop_out_rate_C: dict

@dataclass
class Transition():
    J_u: float
    J_l: float
    Einstein_A: u.Quantity
    Freq: u.Quantity
    E_u: u.Quantity
    tau_0: float
    beta_escape: float
    T_ex: u.Quantity
    T_ant: u.Quantity # T_R in RADEX, CMB background subtracted
    P_lu: u.Quantity
    P_ul: u.Quantity
    I_nu_t: u.Quantity # internal radiation
    I_nu_bkg: u.Quantity # internal background
    I_nu_CMB: u.Quantity # observation-like background
    I_nu_tau0: u.Quantity # observation-like radiation, background subtracted

@dataclass
class Collision():
    J_u: float
    J_l: float
    rate_spl_x: list
    rate_spl_y: list
    # rate_spl: object
    # rate: object
    
    def __post_init__(self):
        # self.rate_spl = UnivariateSpline(self.rate_spl_x, self.rate_spl_y)
        # self.rate_spl.set_smoothing_factor(0.5) # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.UnivariateSpline.html#scipy.interpolate.UnivariateSpline
        # --> UnivariateSpline is too inaccurate
        self.rate_spl = interp1d(self.rate_spl_x, self.rate_spl_y)
        # rate = lambda T: rate_spl(T) * u.cm**3 / u.s
    
    def rate(self, T):
        if isinstance(T, u.Quantity):
            T_val = T.cgs.value
        else:
            T_val = T
        # return self.rate_spl(T_val) * u.cm**3 / u.s
        return 10**(self.rate_spl(np.log10(T_val))) * u.cm**3 / u.s

# blackbody_nu = lambda nu, T: (BlackBody(temperature=T))(nu)
def blackbody_nu(nu, T):
    if T > 0.0 * u.K:
        return (BlackBody(temperature=T))(nu)
    else:
        return 0.0 * u.erg * u.s**(-1) * u.cm**(-2) * u.Hz**(-1) * u.sr**(-1)

c = const.c.cgs
h = const.h.cgs
k = const.k_B.cgs

T_CMB0 = 2.73 * u.K

pprinter = PrettyPrinter(width=200, compact=True)


class AtomMoleculeData():
    """docstring for AtomMoleculeData"""
    def __init__(self):
        self.data_file = ''
        self.species = ''
        self.levels = {}
        self.transitions = {}
        self.collisions = {}
        self.collision_partners = {}
        self.rate_matrix = None
        self.species_column_density = None
        self.line_width = None
        self.T_kin = None
        self.collision_partner_densities = None
        self.solved_column_densities = None
        self.LTE_column_densities = None
        # 
        self.T_dust = None
        self.eta_dust = None
        self.tau_dust = None
        self.z = None
        # 
        self.converged_iterations = 0 # 0 means no iteration, positive means converged, negative means not converged
    
    def __str__(self):
        this_strlist = []
        this_strlist.append('Species: {}'.format(self.species))
        this_screenwidth = 100
        # 
        this_strhead = '  Levels: '
        this_strline = this_strhead
        n_levels = 0
        for i, level in enumerate(list(self.levels.keys())):
            n_levels += 1
        this_strline += '{} levels'.format(n_levels)
        this_strlist.append(this_strline)
        # 
        this_strline = ' '*len(this_strhead)
        for i, level in enumerate(list(self.levels.keys())):
            this_str = '' if i > 0 else 'J='
            this_str += '%g'%(self.levels[level].J)
            if len(this_strline + this_str) > this_screenwidth:
                this_strlist.append(this_strline)
                this_strline = ' '*len(this_strhead)
            elif i > 0:
                this_strline += ' '
            this_strline += this_str
            if i == len(self.levels) and this_strline.strip() != '':
                this_strlist.append(this_strline)
                this_strline = ' '*len(this_strhead)
        # 
        this_strhead = '  Transitions: '
        this_strline = this_strhead
        n_radiation_transition = 0
        for i, up in enumerate(list(self.transitions.keys())):
            for j, low in enumerate(list(self.transitions[up].keys())):
                n_radiation_transition += 1
        this_strline += '{} transitions'.format(n_radiation_transition)
        this_strlist.append(this_strline)
        # 
        # this_strline = ' '*len(this_strhead)
        # for i, up in enumerate(list(self.transitions.keys())):
        #     for j, low in enumerate(list(self.transitions[up].keys())):
        #         this_str = '' if i > 0 else 'J='
        #         this_str += '%g-%g'%(self.transitions[up][low].J_u, self.transitions[up][low].J_l)
        #         if len(this_strline + this_str) > this_screenwidth:
        #             this_strlist.append(this_strline)
        #             this_strline = ' '*len(this_strhead)
        #         elif i > 0:
        #             this_strline += ' '
        #         this_strline += this_str
        #     if i == len(self.transitions) and this_strline.strip() != '':
        #         this_strlist.append(this_strline)
        #         this_strline = ' '*len(this_strhead)
        # 
        this_strhead = '  Collisions: '
        this_strline = this_strhead
        n_collision_partner = 0
        for i, collision_partner in enumerate(list(self.collision_partners.keys())):
            n_collision_partner += 1
        n_collision_transition = 0
        for i, up in enumerate(list(self.collisions.keys())):
            for j, low in enumerate(list(self.collisions[up].keys())):
                n_collision_transition += 1
        this_strline += '{} partners, {} transitions'.format(n_collision_partner, n_collision_transition)
        this_strlist.append(this_strline)
        # 
        # this_strline = ' '*len(this_strhead)
        # for i, up in enumerate(list(self.collisions.keys())):
        #     for j, low in enumerate(list(self.collisions[up].keys())):
        #         this_str = '' if i > 0 else 'J='
        #         this_str += '%g-%g'%(self.collisions[up][low].J_u, self.collisions[up][low].J_l)
        #         if len(this_strline + this_str) > this_screenwidth:
        #             this_strlist.append(this_strline)
        #             this_strline = ' '*len(this_strhead)
        #         elif i > 0:
        #             this_strline += ' '
        #         this_strline += this_str
        #     if i == len(self.collisions) and this_strline.strip() != '':
        #         this_strlist.append(this_strline)
        #         this_strline = ' '*len(this_strhead)
        # 
        # print rate matrix if available
        if self.rate_matrix is not None:
            this_strhead = '  Rate Matrix: '
            this_strline = this_strhead
            this_strline += '({} x {})'.format(self.rate_matrix.shape[0], self.rate_matrix.shape[1])
            n_col = min(10, self.rate_matrix.shape[1])
            n_row = min(10, self.rate_matrix.shape[0])
            if n_col * n_row < self.rate_matrix.shape[1] * self.rate_matrix.shape[0]:
                this_strline += ' (showing {} x {})'.format(n_col, n_row)
            this_strlist.append(this_strline)
            # 
            this_strline = ' '*len(this_strhead)
            for j in range(n_row):
                for i in range(n_col):
                    if i == 0 and j > 0:
                        this_strlist.append(this_strline)
                        this_strline = ' '*len(this_strhead)
                    elif i > 0:
                        this_strline += ' '
                    this_strline += '{:10.3e}'.format(self.rate_matrix[j, i])
            this_strlist.append(this_strline)
        # 
        # print species column density if available
        if self.species_column_density is not None:
            this_strhead = '  Column Density: '
            this_strline = this_strhead
            this_str = '{:9.3e}'.format(self.species_column_density)
            this_strline += this_str
            this_strlist.append(this_strline)
        # 
        # print kinetic temperature if available
        if self.T_kin is not None:
            this_strhead = '  Kinetic Temperature: '
            this_strline = this_strhead
            this_str = '{:g}'.format(self.T_kin)
            this_strline += this_str
            this_strlist.append(this_strline)
        # 
        # print collision partner densities if available
        if self.collision_partner_densities is not None:
            this_strhead = '  Collision Partner Densities: '
            this_strline = this_strhead
            for i, collision_partner in enumerate(list(self.collision_partner_densities.keys())):
                this_str = '"{}": {:9.3e}'.format(collision_partner, self.collision_partner_densities[collision_partner])
                if len(this_strline + this_str) > this_screenwidth:
                    this_strlist.append(this_strline)
                    this_strline = ' '*len(this_strhead)
                elif i > 0:
                    this_strline += ' '
                this_strline += this_str
            this_strlist.append(this_strline)
        # 
        # print solved column densities if available
        if self.solved_column_densities is not None:
            this_strhead = '  Solved Level Populations: '
            this_strline = this_strhead
            for i in range(len(self.solved_column_densities)):
                this_str = '{:9.3e}'.format(self.solved_column_densities[i])
                if len(this_strline + this_str) > this_screenwidth:
                    this_strlist.append(this_strline)
                    this_strline = ' '*len(this_strhead)
                elif i > 0:
                    this_strline += ' '
                this_strline += this_str
            this_strlist.append(this_strline)
        # 
        # print LTE column densities if available
        if self.LTE_column_densities is not None:
            this_strhead = '  LTE Level Populations: '
            this_strline = this_strhead
            for i in range(len(self.LTE_column_densities)):
                this_str = '{:9.3e}'.format(self.LTE_column_densities[i])
                if len(this_strline + this_str) > this_screenwidth:
                    this_strlist.append(this_strline)
                    this_strline = ' '*len(this_strhead)
                elif i > 0:
                    this_strline += ' '
                this_strline += this_str
            this_strlist.append(this_strline)
        # 
        return '\n'.join(this_strlist)
    
    def load_data_file(self, data_file_name, verbose = True):
        data_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', data_file_name)
        if not os.path.isfile(data_file_path):
            raise Exception('Error! File not found: %r'%(data_file_path))
        if verbose:
            print('Loading "{}"'.format(data_file_path))
        self.data_file = data_file_path
        self.levels = {}
        self.transitions = {}
        self.collisions = {}
        self.collision_partners = {}
        with open(self.data_file, 'r') as fp:
            current_mode = ''
            collision_partner = ''
            skip_line = 0
            do_read_line = True
            while do_read_line:
                # 
                line_str = fp.readline()
                if not line_str:
                    do_read_line = False
                    break
                # 
                line_str = line_str.rstrip()
                # 
                if line_str == '!NUMBER OF ENERGY LEVELS':
                    line_str = fp.readline()
                    line_num = int(line_str)
                    line_str = fp.readline() # '!LEVEL + ENERGIES(cm^-1) + WEIGHT + J'
                    for i in range(line_num):
                        line_str = fp.readline()
                        level, energy, weight, J = line_str.strip().split()
                        level, energy, weight, J = int(level), float(energy), float(weight), float(J)
                        self.levels[level] = Level(
                                energy=energy*u.cm**(-1), weight=weight, J=J, 
                                column_density=np.nan*u.cm**(-2), 
                                pop_in_rate_A={}, pop_in_rate_B={}, pop_in_rate_C={}, 
                                pop_out_rate_A={}, pop_out_rate_B={}, pop_out_rate_C={}, 
                            )
                        # energy = Freq.cgs/c.cgs, in units of cm^{-1}.
                # 
                elif line_str == '!NUMBER OF RADIATIVE TRANSITIONS':
                    line_str = fp.readline()
                    line_num = int(line_str)
                    line_str = fp.readline() # '!TRANS + UP + LOW + EINSTEINA(s^-1) + FREQ(GHz) + E_u(K)'
                    for i in range(line_num):
                        line_str = fp.readline()
                        trans, up, low, Einstein_A, Freq, E_u = line_str.strip().split()
                        trans, up, low, Einstein_A, Freq, E_u = int(trans), int(up), int(low), float(Einstein_A), float(Freq), float(E_u)
                        J_u = self.levels[up].J
                        J_l = self.levels[low].J
                        if up not in self.transitions.keys():
                            self.transitions[up] = {}
                        self.transitions[up][low] = Transition(
                                J_u=J_u, J_l=J_l, Einstein_A=Einstein_A/u.s, Freq=(Freq*u.GHz).to(u.Hz), E_u=E_u*u.K, 
                                tau_0=np.nan, beta_escape=np.nan, 
                                T_ex=np.nan*u.K, T_ant=np.nan*u.K, 
                                P_lu=np.nan, P_ul=np.nan, 
                                I_nu_t=np.nan, I_nu_bkg=np.nan, I_nu_CMB=np.nan, I_nu_tau0=np.nan, 
                            )
                # 
                elif line_str == '!NUMBER OF COLL PARTNERS':
                    line_str = fp.readline()
                    collision_partner_num = int(line_str)
                    for ii in range(collision_partner_num):
                        line_str = fp.readline() # '!COLLISIONS BETWEEN'
                        line_str = fp.readline()
                        collision_partner = line_str.strip()
                        if re.match(r'.*\b(oH2|o-H2)\b.*', collision_partner):
                            collision_partner = 'ortho-H2'
                        elif re.match(r'.*\b(pH2|p-H2)\b.*', collision_partner):
                            collision_partner = 'para-H2'
                        print('collision_partner: {}'.format(collision_partner))
                        if collision_partner not in self.collision_partners:
                            self.collision_partners[collision_partner] = line_str
                        line_str = fp.readline() # '!NUMBER OF COLL TRANS'
                        line_str = fp.readline()
                        collision_transition_num = int(line_str)
                        line_str = fp.readline() # '!NUMBER OF COLL TEMPS'
                        line_str = fp.readline()
                        collision_temperature_num = int(line_str)
                        line_str = fp.readline() # '!COLL TEMPS'
                        line_str = fp.readline()
                        collision_temperatures = np.array(line_str.strip().split()).astype(float)
                        line_str = fp.readline() # '!TRANS+ UP+ LOW+ COLLRATES(cm^3 s^-1)'
                        for i in range(collision_transition_num):
                            line_str = fp.readline()
                            line_split = line_str.strip().split()
                            trans, up, low = line_split[0:3]
                            trans, up, low = int(trans), int(up), int(low)
                            J_u = self.levels[up].J
                            J_l = self.levels[low].J
                            collision_rates = np.array(line_split[3:]).astype(float)
                            # rate_spl = UnivariateSpline(collision_temperatures, collision_rates)
                            # rate_spl.set_smoothing_factor(0.5) # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.UnivariateSpline.html#scipy.interpolate.UnivariateSpline
                            # rate = lambda T: rate_spl(T) * u.cm**3 / u.s
                            if up not in self.collisions.keys():
                                self.collisions[up] = {}
                            if low not in self.collisions[up].keys():
                                self.collisions[up][low] = {}
                            self.collisions[up][low][collision_partner] = Collision(J_u=J_u, J_l=J_l, 
                                                                            rate_spl_x = np.log10(collision_temperatures), 
                                                                            rate_spl_y = np.log10(collision_rates),
                                                                            # rate_spl=rate_spl, 
                                                                            # rate=rate,
                                                                            )
                            # 
                            # debug print
                            # if up == 2 and low == 1:
                            #     print('debug print in load_data_file')
                            #     print('collision_temperatures: {}'.format(collision_temperatures[0:10]))
                            #     print('collision_rates: {}'.format(collision_rates[0:10]))
                            #     print('rate_spl(50): {}'.format(self.collisions[up][low][collision_partner].rate(50.0)))
                # 
    
    def get_partition_function(self, T, return_array = False):
        """Return partition function Q = sum(g_i * exp(-E_i/(k*T)))
        """
        Q = np.nan
        Q_array = None
        # 
        check_T_dimension = True
        if not isinstance(T, u.Quantity):
            if np.isscalar(T):
                T = T * u.K
            else:
                if len(T) != len(self.levels):
                    check_T_dimension = False
                else:
                    T = u.Quantity([tt * u.K for tt in T])
        else:
            if T.ndim > 0:
                if len(T) != len(self.levels):
                    check_T_dimension = False
        if not check_T_dimension:
            raise Exception('Error! Wrong input array length! len(T) = {}, len(self.levels) = {}'.format(
                            len(T), len(self.levels)))
        # 
        if len(self.levels) > 0:
            Q_array = np.full(len(self.levels), fill_value = 0.0)
            for i, level in enumerate(list(self.levels.keys())):
                g_i = self.levels[level].weight
                if i == 0:
                    Q_array[i] = g_i # g_i * np.exp(-E_i/T)
                else:
                    upper_level = level
                    lower_level = list(self.transitions[level].keys())[0]
                    E_i = self.transitions[upper_level][lower_level].E_u
                    if T.ndim == 0:
                        Q_array[i] = g_i * np.exp(-E_i/T)
                    else:
                        Q_array[i] = g_i * np.exp(-E_i/T[i])
            Q = np.sum(Q_array)
            # print('Q_array', Q_array)
        # 
        if return_array:
            return Q_array
        return Q
    
    def get_dust_opacity_kappa(self, lambda_um, beta = None):
        """Get dust opacity. 
        
        Dust opacity = absorption cross section per unit mass.
        
        Dust optical depth = sigma * integral of dust volumetric density
                           = Q_abs * pi * a**2 * N_dust
        
        See Li & Draine 2001 (2001ApJ...554..778L) Table 6. Adapted the functional form. 
        """
        if beta is None:
            if lambda_um >= 700.0:
                beta = 1.68
            else:
                beta = 2.0
        kappa_ = 0.596 * (lambda_um / 700.0)**(-beta)
        return kappa_ * u.cm**2 / u.g
    
    def get_line_center_optical_depth(self, 
            species_column_density, 
            line_width, 
            T_ex, 
            J_u = None, 
            J_l = None, 
            upper_level = None, 
            lower_level = None, 
            LTE = True, 
            N_u = None, 
            N_l = None, 
        ):
        r"""Return line center optical depth.
        
        By definition:
        
        .. math::
            
            \tau_{0} \equiv \frac{h \nu}{4 \pi} \, \left[ N_{X,l} B_{lu} - N_{X,u} B_{ul} \right] \, \phi_{ul}(0)
        
        With Einstein A B coefficient relation: 
            
            B_{ij} \equiv \frac{c^2}{2 h \nu^3} A_{ij} \\
            
            \tau_{0} = \frac{c^2}{8 \pi \nu_{ul}^{2}} \, \left[ N_{X,l} A_{lu} - N_{X,u} A_{ul} \right] \, \phi_{ul}(0)
        
        With Boxcat line profile: 
            
            \phi_{ul}(0) = 1 / {\Delta \nu} = c / (\nu_{0} \Delta v) \\
            
            \tau_{0} = \frac{c^3}{8 \pi \nu_{ul}^{3}} \, \frac{1}{\Delta \nu} \, \left[ N_{X,l} A_{lu} - N_{X,u} A_{ul} \right]
        
        Or with Gaussian line profile: 
            
            \phi_{ul}(0) = 1 / (\sqrt{2 \pi} / \sqrt{8 \ln 2} \times {\Delta \nu}) = c / (\sqrt{\pi / (4 \ln 2)} \nu_{0} \Delta v) = c / (1.0645 \nu_{0} \Delta v) \\
            
            \tau_{0} = \frac{c^3}{8 \pi \nu_{ul}^{3}} \, \frac{1}{\Delta \nu} \, \left[ N_{X,l} A_{lu} - N_{X,u} A_{ul} \right] \\
            
            \tau_{0} = \frac{c^3}{8 \pi \nu_{ul}^{3}} \, \frac{1}{1.0645 \Delta \nu} \, \left[ N_{X,l} A_{lu} - N_{X,u} A_{ul} \right]
        
        With detailed balance: 
        
        .. math::
            
            g_{i} B_{ij} = g_{j} B_{ji} \\
            
            \tau_{0} = \frac{c^3}{8 \pi \nu_{ul}^{3}} \frac{g_{u}}{g_{l}} A_{ul} \frac{N_{{X,l}}}{1.0645 \Delta v} \left[ 1 - \frac{g_{l} N_{{X,u}}}{g_{u} N_{{X,l}}} \right]
        
        Under LTE: 
        
        .. math::
            
            \frac{g_{l} N_{{X,u}}}{g_{u} N_{{X,l}}} = \exp\left(\frac{-h \nu_{ul}}{k_{\mathrm{B}} T_{\mathrm{ex}}}\right)
            
            \tau_{0} = \frac{c^3}{8 \pi \nu_{ul}^{3}} \frac{g_{u}}{g_{l}} A_{ul} \frac{N_{{X,l}}}{1.0645 \Delta v} \left[ 1 - \exp\left(\frac{-h \nu_{ul}}{k_{\mathrm{B}} T_{\mathrm{ex}}}\right) \right]
        
        """
        if len(self.levels) == 0:
            return np.nan
        if len(self.transitions) == 0:
            return np.nan
        # 
        if J_u is not None and J_l is not None:
            J_u = float(J_u)
            J_l = float(J_l)
            upper_level = [level for level in self.levels.keys() if np.isclose(self.levels[level].J, J_u, rtol=0.0, atol=1e-6)]
            lower_level = [level for level in self.levels.keys() if np.isclose(self.levels[level].J, J_l, rtol=0.0, atol=1e-6)]
            if len(upper_level) == 0:
                raise Exception('Error! Could not find J_u {} in species levels.'.format(J_u))
            if len(lower_level) == 0:
                raise Exception('Error! Could not find J_u {} in species levels.'.format(J_l))
            upper_level = upper_level[0]
            lower_level = lower_level[0]
        elif upper_level is not None and lower_level is not None:
            J_u = self.levels[upper_level].J
            J_l = self.levels[lower_level].J
        else:
            raise Exception('Error! J_u J_l or upper_level lower_level should not all be None!')
        # 
        if not isinstance(T_ex, u.Quantity):
            T_ex = T_ex * u.K
        else:
            T_ex = T_ex.cgs
        # 
        if not isinstance(species_column_density, u.Quantity):
            species_column_density = species_column_density * u.cm**(-2)
        else:
            species_column_density = species_column_density.cgs
        # 
        if not isinstance(line_width, u.Quantity):
            line_width = (line_width * u.km / u.s).cgs
        else:
            line_width = line_width.cgs
        # 
        global c
        global h
        global k
        global T_CMB0
        # 
        nu = self.transitions[upper_level][lower_level].Freq
        g_u = self.levels[upper_level].weight
        g_l = self.levels[lower_level].weight
        A_ul = self.transitions[upper_level][lower_level].Einstein_A
        E_u = self.transitions[upper_level][lower_level].E_u
        if self.levels[lower_level].energy.value > 0.0:
            E_l = self.transitions[lower_level][list(self.transitions[lower_level].keys())[0]].E_u
        else:
            E_l = 0.0 * u.K
        # 
        if LTE:
            Q = self.get_partition_function(T_ex)
            tau_0 = c**3 / (8.0*np.pi*nu**3) * (g_u / Q) * A_ul * species_column_density / (1.0645 * line_width) * \
                    (1.0 - np.exp(-(h * nu) / (k * T_ex))) * \
                    np.exp(E_l / (k * T_ex)) * \
                    (u.Hz**0 * u.s**0)
                    # (1.0 - np.exp(-(h * nu) / (k * T_ex))) goes with g_u 
                    # because [ g_u * (1.0 - np.exp(-(h * nu) / (k * T_ex))) ] / Q == N_u / species_column_density
        else:
            # non-LTE, need population levels
            # 
            # T_ex by definition is:
            N_u_to_N_l = g_u / g_l * np.exp(-(E_u - E_l) / T_ex)
            # 
            # check population levels
            if N_u is None and N_l is None:
                raise Exception('Error! N_u and N_l should not be None when LTE is False!')
            elif N_u is not None:
                N_l = N_u / N_u_to_N_l
            else:
                N_u = N_l * N_u_to_N_l
            # 
            # Lequex 2005 Eq. 3.14
            # -- Note again that (3.12), (3.13) and (3.14) are general despite the assumption of LTE.
            tau_0 = c**3 / (8.0*np.pi*nu**3) * (g_u / g_l) * A_ul * N_l / (1.0645 * line_width) * \
                    (1.0 - (g_l * N_u) / (g_u * N_l) ) * \
                    (u.Hz**0 * u.s**0)
            # 
            # if self.solve_nonLTE(
            #         species_column_density = species_column_density,
            #         line_width = line_width,
            #         T_ex = T_ex,
            #         z = z,
            #         T_dust = T_dust,
            #         tau_dust = tau_dust,
            #         eta_dust = eta_dust,
            #     ):
            #     tau_0 = self.transitions[upper_level][lower_level].tau_0
            # 
        # 
        return tau_0
    
    def get_escape_probability(self, tau, method=''):
        """Escape probability for a uniform sphere from Osterbrock 1974, Astrophysics of Gaseous Nebulae. 
        
        See also van der Tak 2007 RADEX. 
        
        """
        if method == 'expanding_sphere' or method == 'LVG' or method == '':
            # beta_escape = (1.0 - np.exp(-tau)) / tau
            # 
            # if using RADEX method.eq.2 
            # matrix.f
            # c     Expanding sphere = Large Velocity Gradient (LVG) or Sobolev case.
            # C     Formula from De Jong, Boland and Dalgarno (1980, A&A 91, 68)
            # C     corrected by factor 2 in order to match ESCPROB(TAU=0)=1
            if tau < 0.01:
                beta_escape = 1.0
            elif tau < 7.0:
                beta_escape = 2.0*(1.0 - np.exp(-2.34*tau))/(4.68*tau)
            else:
                beta_escape = 2.0/(tau*4.0*(np.sqrt(np.log(tau/np.sqrt(np.pi)))))
        elif method == 'uniform_sphere':
            beta_escape = (1.5 / tau) * (1.0 - (2.0/tau**2) + (2.0/tau + 2.0/tau**2) * np.exp(-tau))
        elif method == 'homogeneous_slab':
            beta_escape = (1.0 - np.exp(-3.0*tau)) / (3.0*tau)
        elif method == 'turbulent_medium':
            beta_escape = 1.0 / (tau * np.sqrt(np.pi * np.log(tau/2.0)))
        else:
            raise Exception('Error! get_escape_probability has a wrong method input!' +
                            'Please set method = \'expanding_sphere\' (default), \'uniform_sphere\', \'homogeneous_slab\' or \'homogeneous_slab\'.')
        return beta_escape
    
    def evaluate_level_populations(self, 
            species_column_density = None, 
            line_width = None, 
            T_kin = None, 
            collision_partner_densities = None, 
            T_dust = None,
            tau_dust = None,
            eta_dust = None, 
            z = None, 
            LTE = True, 
            verbose = False, 
            silent = False, 
        ):
        """TBD
        
        Args:
            collision_partner_densities: dict. Example: `{'ortho-H2': 0.75*1e4, 'para-H2': 0.25*1e4}`.
        
        """
        # 
        if len(self.levels) == 0:
            print('Error! self.levels has no element!')
            return None
        if len(self.transitions) == 0:
            print('Error! self.transitions has no element!')
            return None
        # 
        if verbose:
            silent = False
        # 
        # check input error
        has_error = False
        # 
        if species_column_density is not None:
            if not isinstance(species_column_density, u.Quantity):
                species_column_density = species_column_density * u.cm**(-2)
            else:
                species_column_density = species_column_density.cgs
            if not silent:
                print('Using input species_column_density {}'.format(species_column_density))
        else:
            if self.species_column_density is not None:
                if not silent:
                    print('Using self.species_column_density {}'.format(self.species_column_density))
                species_column_density = copy.copy(self.species_column_density)
            else:
                print('Error! Please input species_column_density!')
                has_error = True
        # 
        if line_width is not None:
            if not isinstance(line_width, u.Quantity):
                line_width = (line_width * u.km/u.s).cgs
            else:
                line_width = line_width.cgs
            if not silent:
                print('Using input line_width {}'.format(line_width))
        else:
            if self.line_width is not None:
                if not silent:
                    print('Using self.line_width {}'.format(self.line_width))
                line_width = copy.copy(self.line_width)
            else:
                print('Error! Please input line_width!')
                has_error = True
        # 
        if T_kin is not None:
            if not isinstance(T_kin, u.Quantity):
                T_kin = T_kin * u.K
            else:
                T_kin = T_kin.cgs
            if not silent:
                print('Using input T_kin {}'.format(T_kin))
        else:
            if self.T_kin is not None:
                if not silent:
                    print('Using self.T_kin {}'.format(self.T_kin))
                T_kin = copy.copy(self.T_kin)
            else:
                print('Error! Please input T_kin!')
                has_error = True
        # 
        if collision_partner_densities is not None:
            for i, collision_partner in enumerate(list(collision_partner_densities.keys())):
                if not isinstance(collision_partner_densities[collision_partner], u.Quantity):
                    collision_partner_densities[collision_partner] *= u.cm**(-3)
            if not isinstance(collision_partner_densities, (dict, OrderedDict)):
                raise Exception('Error! collision_partner_densities should be a dict! Example: {\'ortho-H2\': 0.75e4, \'para-H2\': 0.25*1e4}')
            if not silent:
                print('Using input collision_partner_densities {}'.format(str(collision_partner_densities)))
        else:
            if self.collision_partner_densities is not None:
                if not silent:
                    print('Using self.collision_partner_densities {}'.format(str(self.collision_partner_densities)))
                collision_partner_densities = copy.copy(self.collision_partner_densities)
            else:
                print('Warning! collision_partner_densities is None! Please provide a dict if collisions need to be taken into account!')
        # 
        if T_dust is not None:
            if not isinstance(T_dust, u.Quantity):
                T_dust = T_dust * u.K
            else:
                T_dust = T_dust.cgs
            if not silent:
                print('Using input T_dust {}'.format(T_dust))
        else:
            if self.T_dust is not None:
                if not silent:
                    print('Using self.T_dust {}'.format(self.T_dust))
                T_dust = self.T_dust
            else: 
                T_dust = 30.0 * u.K
                if not silent:
                    print('Using default T_dust {}'.format(T_dust))
        # 
        if eta_dust is not None:
            pass
            if not silent:
                print('Using input T_dust {}'.format(eta_dust))
        else:
            if self.eta_dust is not None:
                if not silent:
                    print('Using self.eta_dust {}'.format(self.eta_dust))
                eta_dust = self.eta_dust
            else: 
                eta_dust = 0.0
                if not silent:
                    print('Using default eta_dust {}'.format(eta_dust))
        # 
        if tau_dust is not None:
            pass
            if not silent:
                print('Using input tau_dust {}'.format(tau_dust))
        else:
            if self.tau_dust is not None:
                if not silent:
                    print('Using self.tau_dust {}'.format(self.tau_dust))
                tau_dust = self.tau_dust
            else: 
                tau_dust = None
                if not silent:
                    print('Using default tau_dust {}'.format(tau_dust))
        # 
        if z is not None:
            pass
            if not silent:
                print('Using input z {}'.format(z))
        else:
            if self.z is not None:
                if not silent:
                    print('Using self.z {}'.format(self.z))
                z = self.z
            else: 
                z = 0.0
                if not silent:
                    print('Using default z {}'.format(z))
        # 
        if has_error:
            return None
        # 
        self.species_column_density = copy.copy(species_column_density)
        self.line_width = copy.copy(line_width)
        self.T_kin = copy.copy(T_kin)
        self.collision_partner_densities = copy.copy(collision_partner_densities)
        self.T_dust = T_dust
        # 
        if LTE:
            if not silent:
                print('Using LTE assumption')
        elif np.any(np.isnan([self.levels[level].column_density.value for level in self.levels.keys()])):
            print('Column densities: {}'.format([self.levels[level].column_density.value for level in self.levels.keys()]))
            raise Exception('When LTE is False, please set self.levels[level].column_density!')
        else:
            if not silent:
                print('Using non-LTE assumption')
        # 
        global c
        global h
        global k
        global T_CMB0
        T_CMB = T_CMB0 * (1.0 + z)
        DGR = 100.0 # dust-to-gas ratio, not used because we do not consider dust emission's contribution to the source function
        # 
        Q = self.get_partition_function(T_kin)
        # 
        if LTE: 
            self.LTE_column_densities = np.zeros(len(self.levels))
        # 
        out_delta_N_square = 0.0
        # 
        # reset rate matrix
        self.rate_matrix = None
        # 
        # I_nu_CMB_dict
        I_nu_CMB_dict = {}
        # 
        for iul, upper_level in enumerate(self.transitions.keys()):
            for ill, lower_level in enumerate(self.transitions[upper_level].keys()):
                nu = self.transitions[upper_level][lower_level].Freq
                g_u = self.levels[upper_level].weight
                g_l = self.levels[lower_level].weight
                J_u = self.levels[upper_level].J
                J_l = self.levels[lower_level].J
                A_ul = self.transitions[upper_level][lower_level].Einstein_A
                # C_ul = self.collisions[upper_level][lower_level].rate(T_kin) * collision_partner_density
                E_u = self.transitions[upper_level][lower_level].E_u
                if self.levels[lower_level].energy > 0.0:
                    E_l = self.transitions[lower_level][list(self.transitions[lower_level].keys())[0]].E_u
                else:
                    E_l = 0.0 * u.K
                # 
                # collision partners
                if collision_partner_densities is not None:
                    C_ul = 0.0 / u.s
                    for partner in collision_partner_densities.keys():
                        partner_density = collision_partner_densities[partner]
                        # if not isinstance(partner_density, u.Quantity):
                        #     partner_density = partner_density * u.cm**(-3)
                        C_ul += self.collisions[upper_level][lower_level][partner].rate(T_kin) * partner_density
                        # 
                        # debug print
                        # if upper_level == 2 and lower_level == 1:
                        #     print('debug print in evaluate_level_populations')
                        #     print('T_kin = {}, partner = "{}"'.format(T_kin, partner))
                        #     print('crate = {}'.format(self.collisions[upper_level][lower_level][partner].rate(T_kin)))
                        #     print('C_ul += {}'.format(self.collisions[upper_level][lower_level][partner].rate(T_kin) * partner_density))
                else:
                    C_ul = 0.0 / u.s
                # 
                # 
                # Detailed balance for collision
                # Lequex 2005 Eq. 3.38
                # -- The latter expression contains only atomic parameters and remains valid in the general 
                #    case provided that a kinetic temperature TK can be defined.
                C_lu = C_ul * g_u / g_l * np.exp(-h*nu/(k*T_kin))
                # 
                # initial LTE guess
                if LTE:
                    self.levels[upper_level].column_density = g_u * np.exp(-E_u/T_kin) / Q * copy.copy(species_column_density)
                    self.levels[lower_level].column_density = g_l * np.exp(-E_l/T_kin) / Q * copy.copy(species_column_density)
                    # self.transitions[upper_level][lower_level].T_ex = copy.copy(T_kin)
                    # tau_0 = self.get_line_center_optical_depth(
                    #         species_column_density = species_column_density, 
                    #         line_width = line_width, 
                    #         T_ex = T_ex, 
                    #         upper_level = upper_level, 
                    #         lower_level = lower_level, 
                    #         LTE = True, 
                    #     )
                # 
                # get N_u N_l
                N_u = copy.copy(self.levels[upper_level].column_density)
                N_l = copy.copy(self.levels[lower_level].column_density)
                # overflow_column_density = 1e-50 * u.cm**(-2)
                # if N_u < overflow_column_density:
                #     N_u = overflow_column_density
                # if N_l < overflow_column_density:
                #     N_l = overflow_column_density
                if verbose:
                    print('J_u {}, J_l {}, g_u {:g}, g_l {:g}, Q {:g}'.format(J_u, J_l, g_u, g_l, Q))
                if verbose:
                    print('J_u {}, J_l {}, N_u {:e}, N_l {:e}'.format(J_u, J_l, N_u, N_l))
                if verbose:
                    print('J_u {}, J_l {}, N_u / N_l {:g}, g_u / g_l * np.exp(-h*nu/(k*T_kin)) {:g}'.format(J_u, J_l, 
                        N_u / N_l, g_u / g_l * np.exp(-h*nu/(k*T_kin))
                        ))
                # 
                # compute T_ex
                # T_ex by definition has: 
                # N_u / N_l = g_u / g_l * np.exp(-(E_u - E_l) / T_ex)
                # T_ex = (E_u - E_l) / np.log((g_u / g_l) * (N_l / N_u))
                if LTE:
                    T_ex = T_kin
                else:
                    if N_l <= 0.0 or N_u <= 0.0:
                        T_ex = 0.0 * u.K
                    else:
                        T_ex = (E_u - E_l) / np.log((g_u / g_l) * (N_l / N_u))
                # if verbose:
                #     print('J_u {}, J_l {}, E_u - E_l {:e}, np.log((g_u / g_l) * (N_l / N_u)) {:e}'.format(J_u, J_l, E_u - E_l, 
                #         np.log((g_u / g_l) * (N_l / N_u))
                #         ))
                # 
                # compute tau_0
                tau_0 = self.get_line_center_optical_depth(
                        species_column_density = species_column_density, 
                        line_width = line_width, 
                        T_ex = T_ex, 
                        upper_level = upper_level, 
                        lower_level = lower_level, 
                        LTE = False, 
                        N_l = N_l, 
                    ) # no need to set LTE here because if LTE is set N_l is already LTE value
                # 
                if verbose:
                    print('J_u {}, J_l {}, N_u {:e}, N_l {:e}'.format(J_u, J_l, N_u, N_l))
                    print('J_u {}, J_l {}, T_kin {}, T_ex {}, tau_0 {}'.format(J_u, J_l, T_kin, T_ex, tau_0))
                # 
                # background radiation (bkg): CMB + dust
                # 
                # background radiation (bkg): CMB + dust
                if tau_dust is None:
                    dust_mass_column_density = (10. * DGR * u.M_sun / u.pc**2).to(u.g / u.cm**2) #<TODO># gas surface density 10 Mun pc-2 ?
                    tau_dust = self.get_dust_opacity_kappa((c/nu).to(u.um).value) * dust_mass_column_density
                # print('tau_dust', tau_dust)
                #I_nu_CMB = blackbody_nu(nu, T_CMB)
                if not (upper_level in I_nu_CMB_dict):
                    I_nu_CMB_dict[upper_level] = {}
                if not (lower_level in I_nu_CMB_dict[upper_level]):
                    I_nu_CMB_dict[upper_level][lower_level] = blackbody_nu(nu, T_CMB)
                I_nu_CMB = I_nu_CMB_dict[upper_level][lower_level]
                I_nu_bkg = I_nu_CMB
                if eta_dust > 0.0:
                    I_nu_bkg = I_nu_CMB + \
                               eta_dust * blackbody_nu(nu, T_dust) * (1.0 - np.exp(-tau_dust))
                # 
                # Einstein A B relation
                # Lequex 2005 Eq. 3.12 3.13
                # -- Note again that (3.12), (3.13) and (3.14) are general despite the assumption of LTE.
                B_ul = A_ul / (2.0 * h * nu**3 / c**2) * u.Hz**3 * u.s**3
                if verbose:
                    print('J_u {}, J_l {}, A_ul {}, B_ul {}, C_ul {}'.format(J_u, J_l, A_ul, B_ul, C_ul))
                B_lu = g_u * B_ul / g_l
                # 
                # source function:
                # source function S_nu by definition is:
                # S_nu \equiv j_nu / kappa_nu = - n_u A_ul / (n_u B_ul - n_l B_lu)
                ###S_nu = - (N_u * A_ul) / (N_u * B_ul - N_l * B_lu)
                # 
                # S_nu equals the Planck function of T_ex because of T_ex's definition. 
                # S_nu = - (N_u * A_ul) / (N_u * B_ul - N_l * B_lu)
                #      = - (N_u * A_ul) / ((N_u - N_l * g_u / g_l) * B_ul)
                #      = - 2 h nu^3 / c^2 * 1 / (1 - (N_l / N_u) * (g_u / g_l))
                #      = 2 h nu^3 / c^2 * 1 / ((N_l / N_u) * (g_u / g_l) - 1)
                # T_ex by definition has: 
                # N_u / N_l = g_u / g_l * np.exp(-(E_u - E_l) / T_ex)
                # T_ex = (E_u - E_l) / np.log((g_u / g_l) * (N_l / N_u))
                # Thus:
                # S_nu = 2 h nu^3 / c^2 * 1 / (np.exp(h nu / (k T_ex)) - 1)
                S_nu = blackbody_nu(nu, T_ex)
                # 
                # S_nu does not equal Planck function under non-LTE. 
                # S_nu = - (N_u * A_ul) / (N_u * B_ul - N_l * B_lu)
                #      = - (N_u * A_ul) / ((N_u - N_l * g_u / g_l) * B_ul)
                #      = - 2 h nu^3 / c^2 * 1 / (1 - (N_l / N_u) * (g_u / g_l))
                #      = 2 h nu^3 / c^2 * 1 / ((N_l / N_u) * (g_u / g_l) - 1)
                # under LTE, 
                # the Boltzmann distribution gives: (N_u/N_l) = (g_u/g_l) * exp(-h nu/(k T)), 
                # so S_nu becomes the Planck function: 2 h nu^3 /c ^2 * 1 / (exp(h nu/(k T)) - 1)
                # 
                # Sobolev or LVG method with escape probability approximation:
                # S_nu is assumed to be: (1.0 - beta_escape) * B_nu(T_ex)
                # Lequex 2005 Eq. 3.48
                if LTE: 
                    beta_escape = 1.0 # np.exp(-tau_0) # 0.0 # np.exp(-tau_0)
                    # RADEX is using beta=1.0 for the initial LTE condition, 
                    # and I_nu_t is their 'backi(iline)' which is 'backi(iline) = cbi+cmi' in "background.f"
                    # but WHY??? <TODO><20210901><DZLIU>
                else:
                    beta_escape = self.get_escape_probability(tau_0)
                I_nu_t = (1.0 - beta_escape) * S_nu + beta_escape * I_nu_bkg
                if verbose:
                    print('J_u {}, J_l {}, beta_escape {}, I_nu_bkg {}, I_nu_t {}'.format(
                        J_u, J_l, beta_escape, I_nu_bkg, I_nu_t))
                # I_nu_t_LTE = (1.0 - np.exp(-tau_0)) * S_nu + np.exp(-tau_0) * I_nu_bkg
                # if verbose:
                #     print('J_u {}, J_l {}, beta_escape {}, I_nu_bkg {}, I_nu_t {}, I_nu_t_LTE {}'.format(
                #         J_u, J_l, beta_escape, I_nu_bkg, I_nu_t, I_nu_t_LTE))
                # 
                # an important equation for the rates under escape probab radiation field:
                # when I_nu = (1-beta) * S_nu + beta * I_nu^bkg 
                # and S_nu = - (n_u * A_ul) / (n_u * B_ul - n_l * B_lu)
                # n_u * A_ul + (n_u * B_ul - n_l * B_lu) * I_nu
                # = n_u * A_ul + (n_u * B_ul - n_l * B_lu) * [ (1-beta) * S_nu + beta * I_nu^bkg ]
                # = n_u * A_ul + (n_u * B_ul - n_l * B_lu) * [ (1-beta) * (- (n_u * A_ul) / (n_u * B_ul - n_l * B_lu)) + beta * I_nu^bkg ]
                # = n_u * A_ul + (1-beta) * (- (n_u * A_ul)) + (n_u * B_ul - n_l * B_lu) * beta * I_nu^bkg
                # = beta * (n_u * A_ul) + (n_u * B_ul - n_l * B_lu) * beta * I_nu^bkg
                # = 
                # -- see https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/lineradtrans.html#chap-line-transfer
                # -- "The LVG+EscProb method solves at each location the following statistical equilibrium equation ..."
                # 
                # S_nu = blackbody_nu(nu, T_ex)
                # I_nu_t = blackbody_nu(nu, T_ex) * (1.0 - np.exp(-tau_0)) + I_nu_bkg * np.exp(-tau_0)
                # 
                # I_nu_t  = I_nu_t_LTE
                # 
                # intensity seen by outsider in excess of the CMB background:
                #I_nu_CMB = blackbody_nu(nu, T_CMB)
                I_nu_CMB = I_nu_CMB_dict[upper_level][lower_level]
                I_nu_tau0 = (1.0 - np.exp(-tau_0)) * S_nu + np.exp(-tau_0) * I_nu_bkg # this is toti in radex
                # I_nu_tsbkg = I_nu_tau0 - I_nu_CMB
                # 
                # for background temperature we still use Planck function
                T_bkg = h * nu / k / np.log(1.0 / (I_nu_CMB / (2*h*nu**3/c**2)).cgs.value + 1.0) / u.Hz / u.s
                # 
                # for antenna temperature, we use radio definition Rayleigh-Jeans approximated brightness temperature
                # subtracting the background temperature
                # T_ant = I_nu_tau0.cgs.value / (2*k*nu**2/c**2).cgs.value * u.K - T_bkg
                #T_ant = I_nu_tau0.cgs.value / (2*k*nu**2/c**2).cgs.value * u.K \
                #        - I_nu_CMB.cgs.value / (2*k*nu**2/c**2).cgs.value * u.K
                T_ant = (I_nu_tau0.cgs.value - I_nu_CMB.cgs.value) / (2*k*nu**2/c**2).cgs.value * u.K
                if verbose:
                    print('J_u {}, J_l {}, T_ex {}, T_ant {}, T_bkg {}'.format(J_u, J_l, T_ex, T_ant, T_bkg))
                # 
                # 
                # Level population rates should be steady
                # Lequex 2005 Eq. 3.45
                # N_l * (I_nu_t * B_lu + C_lu) = N_u * (A_ul + I_nu_t * B_ul + C_ul)
                # 
                P_lu = (I_nu_t * u.sr * u.Hz * u.s * B_lu + C_lu)
                P_ul = (A_ul + I_nu_t * u.sr * u.Hz * u.s * B_ul + C_ul)
                if verbose:
                    print('J_u {}, J_l {}, P_lu {:e}, P_ul {:e}'.format(J_u, J_l, P_lu, P_ul))
                if verbose:
                    print('J_u {}, J_l {}, N_l * P_lu {:e}, N_u * P_ul {:e}'.format(J_u, J_l, N_l * P_lu, N_u * P_ul))
                # 
                # Column density and rates should have this criterion:
                #   N_l * P_lu = N_u * P_ul
                delta_N = N_u * P_ul / P_lu - N_l
                out_delta_N_square += (delta_N.value)**2
                # 
                # Level population rate matrix
                # Each line should be a linalg equation = 0 
                # See https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/lineradtrans.html#chap-line-transfer
                # Because we want
                #   \sum\limit_{j>i} [ n_j A_{ji} + ( n_j B_{ji} - n_i B_{ij} ) * I_nu ] 
                #   - \sum\limit_{j<i} [ n_i A_{ij} + ( n_i B_{ij} - n_j B_{ji} ) * I_nu ] 
                #   + \sum\limit_{j \ne i} [ n_j C_{ji} - n_i C_{ij} ] 
                #   = 0
                # So for each row, from left to right, cells represent the coefficient multiplied to N_1, N_2, N_3, ...
                # Also consider the transition Delta_J = +- 1, 
                # for row i, the above equation is:
                #    N_{i+1} A_{i+1} + ( N_{i+1} B_{i+1,i} - N_{i} B_{i,i+1} ) * I_nu 
                #    - N_{i} A_{i,i-1} + ( N_{i} B_{i,i-1} - N_{i-1} B_{i-1,i} ) * I_nu 
                #    + \sum\limit_{j \ne i} N_{j} C_{j,i}
                #    - N_{i} * \sum\limit_{j \ne i} C_{j,i}
                # so we need to build a rate matrix, where each row is one such balance equation. 
                # 
                # 
                # assert not np.isnan(A_ul)
                # assert not np.isnan(B_ul)
                # assert not np.isnan(B_lu)
                # assert not np.isnan(I_nu_t.value)
                # 
                # Level population rates
                # Lower level pop in from upper to lower
                self.levels[lower_level].pop_in_rate_A[upper_level] = A_ul
                self.levels[lower_level].pop_in_rate_B[upper_level] = I_nu_t * u.sr * u.Hz * u.s * B_ul
                # self.levels[lower_level].pop_in_rate_C[upper_level] = C_ul
                # 
                # Lower level pop out from lower to upper
                self.levels[lower_level].pop_out_rate_B[upper_level] = I_nu_t * u.sr * u.Hz * u.s * B_lu
                # self.levels[lower_level].pop_out_rate_C[upper_level] = C_lu
                # 
                # Upper level pop in from lower to upper
                # note that there is no spontaneous transition (rate_A) from lower to upper
                self.levels[upper_level].pop_in_rate_B[lower_level] = I_nu_t * u.sr * u.Hz * u.s * B_lu
                # self.levels[upper_level].pop_in_rate_C[lower_level] = C_lu
                # 
                # Upper level pop out from upper to lower
                self.levels[upper_level].pop_out_rate_A[lower_level] = A_ul
                self.levels[upper_level].pop_out_rate_B[lower_level] = I_nu_t * u.sr * u.Hz * u.s * B_ul
                # self.levels[upper_level].pop_out_rate_C[lower_level] = C_ul
                # 
                # update T_ex N_u N_l for the next iteration
                # if iiter < max_iter:
                #     T_ex_old = copy.copy(T_ex)
                #     N_u_old = copy.copy(N_u)
                #     N_l_old = copy.copy(N_l)
                #     delta_N = N_u * P_ul / P_lu - N_l
                #     if delta_N < 0:
                #         T_ex *= 1.05
                #     else:
                #         T_ex /= 1.05
                #     N_u -= delta_N * 0.1
                #     N_l += delta_N * 0.1
                #     if T_ex == 0:
                #         break
                #     if verbose:
                #         print('J_u {}, J_l {}, T_ex {} --> {}, N_u {} --> {}, N_l {} --> {}'.format(
                #             J_u, J_l, T_ex_old, T_ex, N_u_old, N_u, N_l_old, N_l))
                # 
                # store into self.transitions
                self.transitions[upper_level][lower_level].tau_0 = tau_0
                self.transitions[upper_level][lower_level].beta_escape = beta_escape
                self.transitions[upper_level][lower_level].T_ex = copy.copy(T_ex)
                self.transitions[upper_level][lower_level].T_ant = copy.copy(T_ant)
                self.transitions[upper_level][lower_level].P_lu = copy.copy(P_lu)
                self.transitions[upper_level][lower_level].P_ul = copy.copy(P_ul)
                self.transitions[upper_level][lower_level].I_nu_t = copy.copy(I_nu_t)
                self.transitions[upper_level][lower_level].I_nu_bkg = copy.copy(I_nu_bkg)
                self.transitions[upper_level][lower_level].I_nu_CMB = copy.copy(I_nu_CMB)
                self.transitions[upper_level][lower_level].I_nu_tau0 = copy.copy(I_nu_tau0)
                # self.levels[upper_level].column_density = copy.copy(N_u)
                # self.levels[lower_level].column_density = copy.copy(N_l)
                # 
        # 
        # 
        # store into self.LTE_column_densities
        if LTE:
            for i, level in enumerate(list(self.levels.keys())):
                self.LTE_column_densities[i] = self.levels[level].column_density.cgs.value
        # 
        # 
        # now consider collisional (de)excitation transitions
        for iul, upper_level in enumerate(self.collisions.keys()):
            for ill, lower_level in enumerate(self.collisions[upper_level].keys()):
                g_u = self.levels[upper_level].weight
                g_l = self.levels[lower_level].weight
                J_u = self.levels[upper_level].J
                J_l = self.levels[lower_level].J
                if self.levels[upper_level].energy > 0.0:
                    E_u = self.transitions[upper_level][list(self.transitions[upper_level].keys())[0]].E_u
                else:
                    E_u = 0.0 * u.K
                if self.levels[lower_level].energy > 0.0:
                    E_l = self.transitions[lower_level][list(self.transitions[lower_level].keys())[0]].E_u
                else:
                    E_l = 0.0 * u.K
                # 
                # collision partners
                if collision_partner_densities is not None:
                    C_ul = 0.0 / u.s
                    for partner in collision_partner_densities.keys():
                        partner_density = collision_partner_densities[partner]
                        # if not isinstance(partner_density, u.Quantity):
                        #     partner_density = partner_density * u.cm**(-3)
                        C_ul += self.collisions[upper_level][lower_level][partner].rate(T_kin) * partner_density
                else:
                    C_ul = 0.0 / u.s
                # 
                # 
                # Detailed balance for collision
                # Lequex 2005 Eq. 3.38
                # -- The latter expression contains only atomic parameters and remains valid in the general 
                #    case provided that a kinetic temperature TK can be defined.
                C_lu = C_ul * g_u / g_l * np.exp(-(E_u-E_l)/T_kin)
                # 
                # 
                # assert not np.isnan(C_ul)
                # assert not np.isnan(C_lu)
                # 
                # 
                # store into levels
                self.levels[lower_level].pop_in_rate_C[upper_level] = C_ul # this will be multiplied by N_l
                self.levels[lower_level].pop_out_rate_C[upper_level] = C_lu # this will be multiplied by N_l
                self.levels[upper_level].pop_in_rate_C[lower_level] = C_lu # by N_u
                self.levels[upper_level].pop_out_rate_C[lower_level] = C_ul # by N_u
        # 
        return out_delta_N_square
    
    
    def build_rate_matrix(self, 
            reduce_rate_matrix = False, 
            only_radiation = False, 
            only_collision = False, 
            verbose = False, 
            silent = False, 
            debug_print = False, 
        ):
        """Build rate matrix, must run after `evaluate_level_populations`. 
        
        Each row `j` of the rate matrix is a linear algorithm equation to solve for level `j`. 
        From left to right, each column `i` corresponds to the (pop-in - pop-out) rate of level `i` to level `j`. 
        
        Note that our rate matrix cell has a positive value if pop in, whereas radex rate matrix is the opposite. 
        
        Also note that our rate matrix dimension is n_levels x n_levels, whereas radex rate matrix dimension is 
        (n_levels+1) x (n_levels+1). This is because for the total population conversation, we replaced the 
        last row, while radex adds a row and column. (But in the latter case, what is the physical meaning of the 
        added column?)
        
        TODO: The arg `reduce_rate_matrix` is to do something similar as radex. Not tested and no plan to test it.
        
        """
        # 
        if verbose:
            silent = False
        #
        self.rate_matrix = None
        rate_matrix = np.zeros((len(self.levels), len(self.levels)))
        # column_density_array = np.zeros(len(self.levels))
        # 
        global c
        global h
        global k
        #
        if not silent:
            print('building rate matrix')
        #
        if reduce_rate_matrix:
            n_reduce = 0
            n_levels = 0
            for jdx, jlevel in enumerate(self.levels.keys()): 
                n_levels += 1
                if (h * self.levels[jlevel].energy * c <= 10.0 * k * self.T_kin):
                    n_reduce += 1
            if not silent:
                print('reducing rate matrix to {} levels with energy <= 10 k T_kin'.format(n_reduce))
        # 
        for jdx, jlevel in enumerate(self.levels.keys()): 
            if verbose:
                print('jlevel', jlevel, 'jdx', jdx)
            # rate_matrix[jlevel, :] = 0.0
            # column_density_array[jdx] = self.levels[jlevel].column_density.cgs.value
            for idx, ilevel in enumerate(self.levels.keys()): 
                # 
                # pop-in from level-ilevel depends on level-ilevel's column density
                if ilevel in self.levels[jlevel].pop_in_rate_A.keys() and not only_collision:
                    rate_matrix[jdx, idx] += self.levels[jlevel].pop_in_rate_A[ilevel].cgs.value
                if ilevel in self.levels[jlevel].pop_in_rate_B.keys() and not only_collision:
                    rate_matrix[jdx, idx] += self.levels[jlevel].pop_in_rate_B[ilevel].cgs.value
                if ilevel in self.levels[jlevel].pop_in_rate_C.keys() and not only_radiation:
                    rate_matrix[jdx, idx] += self.levels[jlevel].pop_in_rate_C[ilevel].cgs.value
                # 
                # pop-out to level-ilevel depends on level-jlevel's column density
                if ilevel in self.levels[jlevel].pop_out_rate_A.keys() and not only_collision:
                    rate_matrix[jdx, jdx] -= self.levels[jlevel].pop_out_rate_A[ilevel].cgs.value
                if ilevel in self.levels[jlevel].pop_out_rate_B.keys() and not only_collision:
                    rate_matrix[jdx, jdx] -= self.levels[jlevel].pop_out_rate_B[ilevel].cgs.value
                if ilevel in self.levels[jlevel].pop_out_rate_C.keys() and not only_radiation:
                    rate_matrix[jdx, jdx] -= self.levels[jlevel].pop_out_rate_C[ilevel].cgs.value
                # 
                # following radex matrix.f lines 199 to 203, "if(eterm(ilev).le.redcrit) nreduce = nreduce+1"
                # we can reduce the rate matrix to speed up calculation,
                # by selecting levels with E_u <= 10 * k T_kin
                #                          h v <= 10 * k T_kin
                if reduce_rate_matrix:
                    if n_reduce < n_levels:
                        for kdx, klevel in enumerate(list(self.levels.keys())[n_reduce:]):
                            # pop-in from ilevel to jlevel passing through klevel
                            rate_matrix[jdx, idx] += (rate_matrix[idx, kdx] * rate_matrix[kdx, jdx]) / rate_matrix[kdx, kdx]
                # if verbose:
                #     print('rate_matrix[{}, {}] = {}'.format(jdx, idx, rate_matrix[jdx, idx]))
        # 
        if reduce_rate_matrix:
            if n_reduce < n_levels:
                rate_matrix_reduced = rate_matrix[0:n_reduce+1, 0:n_reduce+1]
                rate_matrix_reduced[:, -1] = 0.0
                rate_matrix_reduced[-1, :] = 1.0
                rate_matrix = rate_matrix_reduced
                print('reduce_rate_matrix not tested! debug debug')
                #<TODO># not tested!
        # 
        # if verbose:
        #     print('rate_matrix[0:10, 0:10] = ', rate_matrix[0:10, 0:10])
        # 
        # debug print
        if debug_print:
            print('rate_matrix[0:4,0:4] = [')
            rate_matrix_str = ''
            for jjj in range(4):
                rate_matrix_str += '  ['
                for iii in range(4):
                    if iii > 0:
                        rate_matrix_str += ', '
                    rate_matrix_str += '{:6.3E}'.format(rate_matrix[jjj, iii])
                rate_matrix_str += '], # row {}\n'.format(jjj+1)
            rate_matrix_str += ']'
            print(rate_matrix_str)
        # 
        if not silent:
            print('built rate matrix')
        # 
        self.rate_matrix = rate_matrix
        return rate_matrix
    
    
    def solve_rate_matrix(self, 
            # species_column_density = None, 
            store_results = True, 
            verbose = False, 
            silent = False, 
        ):
        """Solve rate matrix with `numpy.linalg.solve`, must run after `evaluate_level_populations` and `build_rate_matrix`. 
        """
        # 
        if verbose:
            silent = False
        # 
        do_solve = True
        if self.rate_matrix is None:
            do_solve = False
        # 
        # if not isinstance(species_column_density, u.Quantity):
        #     species_column_density = species_column_density * u.cm**(-2)
        # else:
        #     species_column_density = species_column_density.cgs
        # 
        # check self.species_column_density
        if self.species_column_density is None:
            print('Error! self.species_column_density is None! Please run `evaluate_level_populations` and `build_rate_matrix` first!')
            do_solve = False
        # 
        solved_column_densities = None
        if do_solve:
            if not silent:
                print('solving rate matrix')
            species_column_density = self.species_column_density
            # 
            # solve TODO 20210830 20h53m
            # rate_matrix = copy.copy(self.rate_matrix)
            # rate_matrix[-1, :] = 1.0
            # rate_balance = np.zeros(len(self.levels))
            # rate_balance[-1] = species_column_density.cgs.value # replacing last row with N_1 + N_2 + ... = N_total -- probably because n_levels rate balance equationss have a degeneracy
            # 
            # adding one row and cell at the end of self.rate_matrix
            # so that the last row means N_1 + N_2 + ... = N_total
            # while the last column is full of zeros, which is just to make sure the matrix is square
            nlev = len(self.levels)
            rate_matrix = np.zeros((nlev+1, nlev+1))
            rate_matrix[0:nlev, 0:nlev] = self.rate_matrix[:, :]
            rate_matrix[nlev, 0:nlev] = 1.0 # last row
            rate_matrix[0:nlev, nlev] = 1e-50 # last column -- can not set to 0.0 otherwise singular matrix error
            rate_balance = np.zeros(nlev+1)
            rate_balance[-1] = species_column_density.cgs.value
            # 
            if verbose:
                print('rate_matrix = ', rate_matrix)
                # print('rate_matrix = [')
                # rate_matrix_str = ''
                # for jjj in range(rate_matrix.shape[0]):
                #     rate_matrix_str += '  ['
                #     for iii in range(rate_matrix.shape[1]):
                #         if iii > 0:
                #             rate_matrix_str += ', '
                #         rate_matrix_str += '{:6.3E}'.format(rate_matrix[jjj, iii])
                #     rate_matrix_str += '], # row {}\n'.format(jjj+1)
                # rate_matrix_str += ']'
                # print(rate_matrix_str)
                print('rate_balance = ', rate_balance)
            # 
            rate_matrix_rank = np.linalg.matrix_rank(rate_matrix, tol=1e-50)
            if verbose:
                print('rate_matrix.shape = ', rate_matrix.shape)
                print('rate_matrix_rank = ', rate_matrix_rank)
            # 
            solved_column_densities = np.linalg.solve(rate_matrix, rate_balance)
            solved_column_densities = solved_column_densities[0:nlev]
            # 
            if verbose:
                print('solved_column_densities = ', solved_column_densities)
            # 
            if not silent:
                print('solved rate matrix')
            # 
            if store_results:
                for i, level in enumerate(self.levels):
                    self.levels[level].column_density = solved_column_densities[i] * u.cm**(-2)
        # 
        # return
        self.solved_column_densities = solved_column_densities
        return solved_column_densities
    
    
    def solve_rate_matrix_iteratively(self, 
            min_iter = 10, 
            max_iter = 100, 
            verbose = False, 
            silent = False, 
        ):
        # 
        if len(self.transitions) == 0:
            print('Error! solve_rate_matrix_iteratively requires self.transitions but it is None!')
            return None
        if len(self.levels) == 0:
            print('Error! solve_rate_matrix_iteratively requires self.levels but it is None!')
            return None
        # 
        if verbose:
            silent = False
        # 
        if not silent:
            print('Solving rate matrix iteratively')
        # 
        if verbose:
            print('Initializing with LTE condition')
        # 
        self.evaluate_level_populations(
                verbose = False, 
                silent = silent, 
                LTE = True, 
            ) # evaluate first to get tau_0 and T_ex
        # 
        if verbose:
            print('Initialized with LTE condition')
        # 
        i_iter = 0
        converged = False
        self.converged_iterations = 0
        while i_iter < max_iter and not converged:
            # 
            prev_column_densities = copy.copy(self.solved_column_densities)
            prev_T_ex = {}
            for iul, upper_level in enumerate(self.transitions.keys()):
                if upper_level not in prev_T_ex:
                    prev_T_ex[upper_level] = {}
                for ill, lower_level in enumerate(self.transitions[upper_level].keys()):
                    prev_T_ex[upper_level][lower_level] = self.transitions[upper_level][lower_level].T_ex.value
            # 
            if verbose:
                print('Iteration {}'.format(i_iter + 1))
            # 
            # # debug
            # print('Debug rate matrix with only radiative (de)excitation:')
            # self.build_rate_matrix(
            #         verbose = False, 
            #         silent = True, 
            #         only_radiation = True, 
            #         debug_print = True, 
            #     )
            # # 
            # print('Debug rate matrix with only collisional (de)excitation:')
            # self.build_rate_matrix(
            #         verbose = False, 
            #         silent = True, 
            #         only_collision = True, 
            #         debug_print = True, 
            #     )
            # 
            self.build_rate_matrix(
                    verbose = False, 
                    silent = True, 
                    # debug_print = True, 
                )
            # 
            self.solve_rate_matrix(
                    store_results = True, 
                    verbose = False, 
                    silent = True, 
                )
            # 
            solved_column_densities = copy.copy(self.solved_column_densities)
            # 
            if verbose:
                print('Iteration {}, column densities before {}, after {}'.format(
                        i_iter + 1, 
                        prev_column_densities, 
                        solved_column_densities, 
                    ))
            # 
            self.evaluate_level_populations(
                    verbose = False, 
                    silent = True, 
                    LTE = False, 
                ) # evaluate again to get tau_0 and T_ex
            # 
            solved_T_ex = {}
            for iul, upper_level in enumerate(self.transitions.keys()):
                if upper_level not in solved_T_ex:
                    solved_T_ex[upper_level] = {}
                for ill, lower_level in enumerate(self.transitions[upper_level].keys()):
                    solved_T_ex[upper_level][lower_level] = self.transitions[upper_level][lower_level].T_ex.value
            # 
            if verbose:
                print('Iteration {}, T_ex before {}, after {}'.format(
                        i_iter + 1, 
                        prev_T_ex, 
                        solved_T_ex, 
                    ))
            # 
            # check converge on T_ex for optically thick lines
            if i_iter > min_iter:
                check_diff = 0.0
                n_thick = 0
                for iul, upper_level in enumerate(self.transitions.keys()):
                    for ill, lower_level in enumerate(self.transitions[upper_level].keys()):
                        if self.transitions[upper_level][lower_level].tau_0 > 0.01:
                            check_diff += np.abs(
                                (prev_T_ex[upper_level][lower_level] - solved_T_ex[upper_level][lower_level]) 
                                / solved_T_ex[upper_level][lower_level]
                            )
                            n_thick += 1
                if (n_thick <= 0) or (check_diff / float(n_thick) < 1e-6):
                    converged = True
            # 
            if prev_column_densities is not None:
                next_column_densities = 0.3 * solved_column_densities + 0.7 * prev_column_densities
            else:
                next_column_densities = solved_column_densities
            # 
            for idx, level in enumerate(self.levels):
                if next_column_densities[idx] < 1e-50:
                    next_column_densities[idx] = 1e-50
                self.levels[level].column_density = next_column_densities[idx] * u.cm**(-2)
                self.solved_column_densities = next_column_densities
            # 
            i_iter += 1
        # 
        if converged:
            self.converged_iterations = i_iter
            if not silent:
                print('Converged with {} iterations'.format(i_iter))
        else:
            self.converged_iterations = -i_iter
            print('Error! Solving rate matrix has not converged with {} iterations (max {})'.format(i_iter, max_iter))
        # 
        if not silent:
            print('Solved rate matrix iteratively')
    
    
    def get_solved_transition_properties(self, 
            frequency_range = None, 
            print_table = False, 
        ):
        # 
        if len(self.transitions) == 0:
            return None
        # 
        global c
        global h
        global k
        # 
        out_dict = OrderedDict()
        out_dict['Line'] = []
        out_dict['E_u'] = []
        out_dict['Freq'] = []
        out_dict['T_ex'] = []
        out_dict['tau_0'] = []
        out_dict['T_ant'] = []
        out_dict['I_nu_tau0'] = []
        # out_dict['I_nu_CMB'] = []
        out_dict['Pop_u'] = []
        out_dict['Pop_l'] = []
        out_dict['Flux_Kkms'] = []
        out_dict['Flux_ergscmp2'] = []
        for iul, upper_level in enumerate(self.transitions.keys()):
            for ill, lower_level in enumerate(self.transitions[upper_level].keys()):
                E_u = self.transitions[upper_level][lower_level].E_u
                J_u = self.levels[upper_level].J
                J_l = self.levels[lower_level].J
                N_u = self.levels[upper_level].column_density
                N_l = self.levels[lower_level].column_density
                Freq = self.transitions[upper_level][lower_level].Freq
                T_ex = self.transitions[upper_level][lower_level].T_ex
                tau_0 = self.transitions[upper_level][lower_level].tau_0
                T_ant = self.transitions[upper_level][lower_level].T_ant
                # I_nu_t = self.transitions[upper_level][lower_level].I_nu_t
                # I_nu_bkg = self.transitions[upper_level][lower_level].I_nu_bkg
                I_nu_tau0 = self.transitions[upper_level][lower_level].I_nu_tau0
                # I_nu_CMB = self.transitions[upper_level][lower_level].I_nu_CMB
                Flux_Kkms = 1.0645 * self.line_width.to(u.km/u.s) * T_ant
                Flux_ergscmp2 = 1.0645 * 8 * np.pi * k * T_ant * self.line_width.cgs * (Freq/c)**3
                out_dict['Line'].append('%g -- %g'%(J_u, J_l))
                out_dict['E_u'].append(E_u)
                out_dict['Freq'].append(Freq)
                out_dict['T_ex'].append(T_ex)
                out_dict['tau_0'].append(tau_0)
                out_dict['T_ant'].append(T_ant)
                out_dict['I_nu_tau0'].append(I_nu_tau0)
                # out_dict['I_nu_CMB'].append(I_nu_CMB)
                out_dict['Pop_u'].append(N_u / self.species_column_density)
                out_dict['Pop_l'].append(N_l / self.species_column_density)
                out_dict['Flux_Kkms'].append(Flux_Kkms)
                out_dict['Flux_ergscmp2'].append(Flux_ergscmp2)
        out_table = Table(out_dict)
        if print_table:
            out_table['Freq'].format = '%10.3E'
            out_table['T_ex'].format = '%10.3E'
            out_table['tau_0'].format = '%10.3E'
            out_table['T_ant'].format = '%10.3E'
            out_table['I_nu_tau0'].format = '%10.3E'
            # out_table['I_nu_CMB'].format = '%10.3E'
            out_table['Pop_u'].format = '%10.3E'
            out_table['Pop_l'].format = '%10.3E'
            out_table['Flux_Kkms'].format = '%10.3E'
            out_table['Flux_ergscmp2'].format = '%10.3E'
            print(out_table)
        return out_table
    
    
    
    def log_prob(self, 
            p, 
            species_column_density, 
            line_width, 
            T_kin, 
            collision_partner_densities, 
            z, 
            T_dust, 
            tau_dust, 
            eta_dust, 
        ):
        # 
        ipar = 0
        n_levels = len(self.levels)
        for ilevel, level in enumerate(self.levels.keys()):
            if ilevel == 0:
                self.levels[level].column_density = species_column_density - (np.sum(p[0:n_levels]) * u.cm**(-2))
            else:
                self.levels[level].column_density = p[ipar] * u.cm**(-2)
                ipar += 1
        if np.count_nonzero(np.array(p[n_levels:] < 0.0)) > 0:
            return -np.inf
        for iul, upper_level in enumerate(self.transitions.keys()):
            for ill, lower_level in enumerate(self.transitions[upper_level].keys()):
                self.transitions[upper_level][lower_level].T_ex = T_kin
                # self.transitions[upper_level][lower_level].T_ex = p[ipar] * u.K
                # ipar += 1
        out_delta_N_square = self.evaluate_level_populations(
                species_column_density = species_column_density, 
                line_width = line_width, 
                T_kin = T_kin, 
                collision_partner_densities = collision_partner_densities, 
                z = z, 
                T_dust = T_dust, 
                tau_dust = tau_dust, 
                eta_dust = eta_dust, 
                verbose = False, 
            )
        return -0.5 * out_delta_N_square
    
    def solve_level_populations_with_mcmc(self, 
            species_column_density, 
            line_width, 
            T_kin, 
            collision_partner_densities = None, 
            z = 0.0, 
            T_dust = 30.0 * u.K, 
            tau_dust = None, 
            eta_dust = 0.0, 
            n_samples = 50, 
        ): 
        # 
        if not isinstance(species_column_density, u.Quantity):
            species_column_density = species_column_density * u.cm**(-2)
        else:
            species_column_density = species_column_density.cgs
        # 
        if not isinstance(line_width, u.Quantity):
            line_width = (line_width * u.km / u.s).cgs
        else:
            line_width = line_width.cgs
        # 
        if not isinstance(T_kin, u.Quantity):
            T_kin = T_kin * u.K
        else:
            T_kin = T_kin.cgs
        # 
        # 
        # n_params
        n_params = 0
        for ilevel, level in enumerate(self.levels.keys()):
            if ilevel > 0:
                n_params += 1
        # for iul, upper_level in enumerate(self.transitions.keys()):
        #     for ill, lower_level in enumerate(self.transitions[upper_level].keys()):
        #         n_params += 1
        # 
        # n_walkers
        n_walkers = n_params * 2 # need at least twice n_params
        # 
        # n_samples
        # n_samples = 2
        # 
        # init_params
        init_params = np.full((n_walkers, n_params), fill_value=np.nan)
        # 
        out_delta_N_square = self.evaluate_level_populations(
            species_column_density = species_column_density, 
            line_width = line_width, 
            T_kin = T_kin, 
            collision_partner_densities = collision_partner_densities, 
            z = z,
            T_dust = T_dust,
            tau_dust = tau_dust,
            eta_dust = eta_dust,
            LTE = True, 
            verbose = False, 
        )
        # 
        ipar = 0
        for ilevel, level in enumerate(self.levels.keys()):
            if ilevel > 0:
                init_params[:, ipar] = self.levels[level].column_density.value
                ipar += 1
        # for iul, upper_level in enumerate(self.transitions.keys()):
        #     for ill, lower_level in enumerate(self.transitions[upper_level].keys()):
        #         init_params[:, ipar] = self.transitions[upper_level][lower_level].T_ex.value
        #         ipar += 1
        init_params *= np.random.uniform(0.95, 1.05, size=init_params.shape)
        # 
        print('n_params', n_params)
        print('init_params.shape', init_params.shape)
        print('init_params', init_params)
        # 
        sampler = emcee.EnsembleSampler(n_walkers, n_params, 
                                        self.log_prob, 
                                        args=[
                                            species_column_density, 
                                            line_width, 
                                            T_kin, 
                                            collision_partner_densities, 
                                            z, 
                                            T_dust, 
                                            tau_dust, 
                                            eta_dust, 
                                        ])
        # 
        pos, prob, state = sampler.run_mcmc(init_params, n_samples, skip_initial_state_check=True, progress=True)
        # 
        # res = plot(sampler.chain[:,:,0].T, '-', color='k', alpha=0.3)
        # 
        # labels = []
        # for ilevel, level in enumerate(self.levels.keys()):
        #     if ilevel > 0:
        #         labels.append('N_{}'.format(level))
        # for iul, upper_level in enumerate(self.transitions.keys()):
        #     for ill, lower_level in enumerate(self.transitions[upper_level].keys()):
        #         labels.append('T_ex_{}_{}'.format(upper_level, lower_level))
        # 
        # tmp = corner(sampler.flatchain, labels=labels)
        # 
        return sampler




class CO(AtomMoleculeData):
    """docstring for CO"""
    def __init__(self):
        super(CO, self).__init__()
        self.species = 'CO'
        self.load_data_file('co.dat')




class CI(AtomMoleculeData):
    """docstring for CI"""
    def __init__(self):
        super(CI, self).__init__()
        self.species = 'CI'
        self.load_data_file('catom.dat')
        
    













