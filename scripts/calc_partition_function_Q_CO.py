#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, re
import numpy as np
from astropy import units as u
from astropy import constants as const
from collections import OrderedDict
from pprint import pprint, PrettyPrinter
# from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


co_level_freq_weights_J_str = """
# J_u    Einstein_A       Freq_GHz       g_u     g_l   J_l   E_u
    1    7.203e-08     115.2712018       3.0     1.0     0   5.53
    2    6.910e-07     230.5380000       5.0     3.0     1   16.60
    3    2.497e-06     345.7959899       7.0     5.0     2   33.19
    4    6.126e-06     461.0407682       9.0     7.0     3   55.32
    5    1.221e-05     576.2679305      11.0     9.0     4   82.97
    6    2.137e-05     691.4730763      13.0    11.0     5   116.16
    7    3.422e-05     806.6518060      15.0    13.0     6   154.87
    8    5.134e-05     921.7997000      17.0    15.0     7   199.11
    9    7.330e-05    1036.9123930      19.0    17.0     8   248.88
   10    1.006e-04    1151.9854520      21.0    19.0     9   304.16
   11    1.339e-04    1267.0144860      23.0    21.0    10   364.97
   12    1.735e-04    1381.9951050      25.0    23.0    11   431.29
   13    2.200e-04    1496.9229090      27.0    25.0    12   503.13
   14    2.739e-04    1611.7935180      29.0    27.0    13   580.49
   15    3.354e-04    1726.6025057      31.0    29.0    14   663.35
   16    4.050e-04    1841.3455060      33.0    31.0    15   751.72
   17    4.829e-04    1956.0181390      35.0    33.0    16   845.59
   18    5.695e-04    2070.6159930      37.0    35.0    17   944.97
   19    6.650e-04    2185.1346800      39.0    37.0    18   1049.84
   20    7.695e-04    2299.5698420      41.0    39.0    19   1160.20
   21    8.833e-04    2413.9171130      43.0    41.0    20   1276.05
   22    1.006e-03    2528.1720600      45.0    43.0    21   1397.38
   23    1.139e-03    2642.3303459      47.0    45.0    22   1524.19
   24    1.281e-03    2756.3875840      49.0    47.0    23   1656.47
   25    1.432e-03    2870.3394070      51.0    49.0    24   1794.23
   26    1.592e-03    2984.1814550      53.0    51.0    25   1937.44
   27    1.761e-03    3097.9093610      55.0    53.0    26   2086.12
   28    1.940e-03    3211.5187506      57.0    55.0    27   2240.24
   29    2.126e-03    3325.0052827      59.0    57.0    28   2399.82
   30    2.321e-03    3438.3646110      61.0    59.0    29   2564.83
   31    2.524e-03    3551.5923610      63.0    61.0    30   2735.28
   32    2.735e-03    3664.6841800      65.0    63.0    31   2911.15
   33    2.952e-03    3777.6357280      67.0    65.0    32   3092.45
   34    3.175e-03    3890.4427170      69.0    67.0    33   3279.15
   35    3.404e-03    4003.1007876      71.0    69.0    34   3471.27
   36    3.638e-03    4115.6055850      73.0    71.0    35   3668.78
   37    3.878e-03    4227.9527744      75.0    73.0    36   3871.69
   38    4.120e-03    4340.1381120      77.0    75.0    37   4079.98
   39    4.365e-03    4452.1571221      79.0    77.0    38   4293.64
   40    4.613e-03    4564.0056399      81.0    79.0    39   4512.67
""" # from Leiden Atomic and Molecular Database, "co.dat"

h = const.h.cgs
k = const.k_B.cgs

co_level_dict = OrderedDict()
for line_str in co_level_freq_weights_J_str.split('\n'):
    line_str = line_str.strip()
    if line_str == '':
        continue
    elif line_str.startswith('#'):
        line_str = line_str.lstrip('#').strip()
        for colname in line_str.split():
            print('co_level_dict[\''+colname+'\'] = []')
            co_level_dict[colname] = []
    else:
        val_list = line_str.strip().split()
        for colindex, colname in enumerate(list(co_level_dict.keys())):
            print('co_level_dict[\''+colname+'\'].append('+str(val_list[colindex])+')')
            co_level_dict[colname].append(float(val_list[colindex]))

pp = PrettyPrinter(width=200, compact=True)
    
# print('--------------')
# print('co_level_dict:')
# pp.pprint(co_level_dict)

nu = (np.array(co_level_dict['Freq_GHz'])*u.GHz).to(u.Hz)
J_u = np.array(co_level_dict['J_u'])
g_u = np.array(co_level_dict['g_u'])
E_u = np.array(co_level_dict['E_u'])*u.K




def calc_partition_function_Q_CO(
        T = 25.0 * u.K, 
        verbose = False, 
        do_plot = False, 
        return_array = False, 
    ):
    
    global pp
    global nu
    global J_u
    global g_u
    global E_u
    
    Q_u = g_u * np.exp(-E_u/T)
    if verbose:
        print('----')
        print('Q_u:')
        pp.pprint(Q_u)
    
    sum_Q_u = np.sum(Q_u).value
    
    if verbose:
        print('---------')
        print('sum(Q_u):')
        pp.pprint(sum_Q_u)

    if verbose:
        print('------------------')
        print('2 * T / (E_1/k_B):')
        pp.pprint(2 * T / E_u[0])
    
    # make plots
    if do_plot:
        fig = px.line(
                x = J_u, 
                y = Q_u,
                title = 'Q_u vs. J_u', 
                log_y = True, 
              )
        fig.show()
    
    # return Q_u
    if return_array:
        return Q_u, J_u
    else:
        return sum_Q_u




def plot_Q_vs_T_CO():
    
    global E_u
    
    T_array = np.arange(5., 155., 1.)
    Q_array = [calc_partition_function_Q_CO(T = T * u.K) for T in T_array]
    TE1_array = 2 * T_array / E_u[0].value * 0.987
    print(Q_array/TE1_array)
    
    fig = go.Figure()
    
    Q_trace = go.Line(
            x = T_array, 
            y = Q_array,
            name = 'sum(Q_u)', 
        )
    
    TE1_trace = go.Scatter(
            x = T_array, 
            y = TE1_array, 
            name = '2 k T / E_1', 
        )
    
    fig.update_layout(
            title = 'Q vs. T', 
        )
    
    fig.update_xaxes(
            title_text = 'T [K]', 
        )
    
    fig.update_yaxes(
            title_text = 'Q or 2 k T / E_1', 
        )
    
    fig.add_trace(Q_trace)
    fig.add_trace(TE1_trace)
    
    # fig.show()
    
    
    
    fig2 = go.Figure()
    
    trace2a = go.Line(
            x = np.log(np.log(T_array/5.)), 
            y = (Q_array/TE1_array),
            name = 'Q/TE1', 
          )
    
    trace2b = go.Line(
            x = np.arange(-1.7, 1.2, 0.1), 
            y = (np.arange(-1.7, 1.2, 0.1) - 1.0) * 0.1 + 0.988,
            name = 'model', 
          )
    
    fig2.add_trace(trace2a)
    fig2.add_trace(trace2b)
    
    fig2.show()
        




if __name__ == '__main__':
    
    # calc_partition_function_Q_CO(T = 25.0 * u.K, verbose = True, do_plot = True)
    
    plot_Q_vs_T_CO()










