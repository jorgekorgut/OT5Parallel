import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./Analysis/Part2/stats.csv',header=None,names=['version','nbcore','input','runtime'],dtype={
                     'version': str,
                     'nbcore': int,
                     'input' : str,
                     'runtime' : float
                 })

subplot, axis = plt.subplots(1)

input_interation_values = df['input'].unique()

colors_array = ['blue', 'red', 'yellow', 'green', 'black', 'magenta', 'cyan', 'orange', 'brown', 'black', 'black']

for index, input in enumerate(input_interation_values):
    df_plot = df[(df['input'] == input)]
    df_plot = df_plot[df_plot['version'] == "sequential"]
    
    mean_stats = df_plot.groupby(['input','version','nbcore']).mean().reset_index()
    
    axis.plot(mean_stats['nbcore'],
     mean_stats['runtime'],
     linestyle="solid",
     color=colors_array[index])

    axis.set_xlabel('nbcores (number)')
    axis.set_ylabel('runtime (log(seconds))')
    axis.set_yscale('log')

    
    #df_plot = df[(df['input'] == input) & (df['version'] == "reduce")]
    #mean_stats = df_plot.groupby(['input','version','nbcore']).mean().reset_index()
    
    #axis[1].plot(mean_stats['nbcore'], mean_stats['runtime'],linestyle="dashed",color=color_input[input])
    #axis[1].set_yscale('log')
    #axis[1].set_xscale('log')
    #axis[1].scatter(df_plot['nbcore'], df_plot['runtime'],color=color_input[input])

# Scatter plot
for index, input in enumerate(input_interation_values):
    df_plot = df[(df['input'] == input)]
    df_plot = df_plot[df_plot['version'] == "sequential"]

    axis.scatter(df_plot['nbcore'], 
    df_plot['runtime'],
    color=colors_array[index])
    axis.set_xlabel('nbcores (number)')
    axis.set_ylabel('runtime (log(seconds))')
    axis.set_yscale('log')


plt.legend(input_interation_values)
plt.show()