import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Analysis/Part1/stats.csv',header=None,names=['input','nbcore','num_steps','runtime'],dtype={
                     'input': str,
                     'nbcore': int,
                     'num_steps' : int,
                     'runtime' : float
                 })

subplot, axis = plt.subplots(1)

input_interation_values = df['input'].unique()

colors_array = ['blue', 'red', 'yellow', 'green', 'black', 'magenta', 'cyan', 'orange', 'brown', 'black', 'black']

def plotCoresGraph(key_input) :
    for index, input in enumerate(input_interation_values):
        df_plot = df[(df['input'] == input)]
        df_plot = df_plot[df_plot['input'] == key_input]
        
        mean_stats = df_plot.groupby(['input','nbcore']).mean().reset_index()
        
        axis.plot(mean_stats['nbcore'],
        mean_stats['runtime'],
        linestyle="solid",
        color=colors_array[index])

        axis.set_xlabel('nbcores (number)')
        axis.set_ylabel('runtime (log(seconds))')
        axis.set_yscale('log')


    # Scatter plot
    for index, input in enumerate(input_interation_values):
        df_plot = df[(df['input'] == input)]
        df_plot = df_plot[df_plot['input'] == key_input]

        axis.scatter(df_plot['nbcore'], 
        df_plot['runtime'],
        color=colors_array[index])
        axis.set_xlabel('nbcores (number)')
        axis.set_ylabel('runtime (log(seconds))')
        axis.set_yscale('log')
    
    plt.title(key_input)
    plt.legend(input_interation_values)
    plt.show()

def plotCompare(key_input1, key_input2, input_iteration_to_compare) :
    for index, input in enumerate(input_interation_values):
        if input in input_iteration_to_compare :
            df_plot = df[(df['input'] == key_input1)]
            
            mean_stats = df_plot.groupby(['input','nbcore']).mean().reset_index()
            
            axis.plot(mean_stats['nbcore'],
            mean_stats['runtime'],
            linestyle="solid",
            color=colors_array[index])


            df_plot = df[(df['input'] == key_input2)]
            
            mean_stats = df_plot.groupby(['input','nbcore']).mean().reset_index()
            
            axis.plot(mean_stats['nbcore'],
            mean_stats['runtime'],
            linestyle="dashed",
            color=colors_array[index])

            axis.set_xlabel('nbcores (number)')
            axis.set_ylabel('runtime (log(seconds))')
            axis.set_yscale('log')
    #plt.title(input_iteration_to_compare)
    plt.legend([key_input1, key_input2])
    plt.show()

#plotCompare("reduce", "divided", [input_interation_values[-1]])

plotCoresGraph("reduce")

# for num_steps in df['num_steps']:

#     df_plot = df[(df['num_steps'] == int(num_steps))]
#     df_plot = df_plot[df_plot['input'] == "divided"]
    
#     mean_stats = df_plot.groupby(['input','nbcore']).mean().reset_index()
    
#     axis.plot(mean_stats['nbcore'], mean_stats['runtime'],linestyle="solid",color=colors_array[num_steps])

#     axis.scatter(df_plot['nbcore'], df_plot['runtime'],color=colors_array[num_steps])

#     # df_plot = df[(df['num_steps'] == num_steps) & (df['input'] == "reduce")]
#     # mean_stats = df_plot.groupby(['num_steps','input','nbcore']).mean().reset_index()
    
#     # axis[1].plot(mean_stats['nbcore'], mean_stats['runtime'],linestyle="dashed",color=colors_array[num_steps])
#     # axis[1].set_yscale('log')
#     # axis[1].set_xscale('log')
#     # axis[1].scatter(df_plot['nbcore'], df_plot['runtime'],color=colors_array[num_steps])