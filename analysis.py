import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('stats.csv',header=None,names=['version','nbcore','num_steps','runtime'],dtype={
                     'version': str,
                     'nbcore': int,
                     'num_steps' : int,
                     'runtime' : float
                 })

color_num_steps = {1e6 : "blue", 1e8 : "red", 1e10 : "green", 1e12 : "black"}

subplot, axis = plt.subplots(2)

for num_steps in df['num_steps']:

    df_plot = df[(df['num_steps'] == int(num_steps))]
    df_plot = df_plot[df_plot['version'] == "divided"]
    
    mean_stats = df_plot.groupby(['num_steps','version','nbcore']).mean().reset_index()
    
    axis[0].plot(mean_stats['nbcore'], mean_stats['runtime'],linestyle="solid",color=color_num_steps[num_steps])
    axis[0].set_yscale('log')
    axis[0].set_xscale('log')
    axis[0].scatter(df_plot['nbcore'], df_plot['runtime'],color=color_num_steps[num_steps])

    df_plot = df[(df['num_steps'] == num_steps) & (df['version'] == "reduce")]
    mean_stats = df_plot.groupby(['num_steps','version','nbcore']).mean().reset_index()
    
    axis[1].plot(mean_stats['nbcore'], mean_stats['runtime'],linestyle="dashed",color=color_num_steps[num_steps])
    axis[1].set_yscale('log')
    axis[1].set_xscale('log')
    axis[1].scatter(df_plot['nbcore'], df_plot['runtime'],color=color_num_steps[num_steps])
    
plt.legend()
plt.show()