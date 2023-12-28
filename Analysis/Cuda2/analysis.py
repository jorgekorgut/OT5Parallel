import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

import pandas as pd

import warnings
warnings.filterwarnings('ignore')


openMP = pd.read_csv('./Analysis/Part2/stats.csv',header=None,names=['version','nbcore','M', 'N','runtime'],dtype={
                     'version': str,
                     'nbcore': int,
                     'M': int,
                     'N': int,
                     'runtime' : float
                 })

cuda2 = pd.read_csv('./Analysis/Cuda2/stats.csv',header=None,names=['optimizationVersion','M','N','runtime'],dtype={
                     'optimizationVersion': str,
                     'M': int,
                     'N': int,
                     'runtime' : float
                 })

subplot, axis = plt.subplots(1)

input_interation_values = cuda2['optimizationVersion'].unique()

colors_array = ['blue', 'red', 'yellow', 'green', 'black', 'magenta', 'cyan', 'orange', 'brown', 'black', 'black']



def plotGraph(df, title) :
    for index, input in enumerate(input_interation_values):
        df_plot = df #[df['optimizationVersion'] == key_input]
        
        mean_stats = df_plot.groupby(['optimizationVersion','M','N']).mean().reset_index()
        mean_stats = mean_stats[mean_stats['optimizationVersion'] == input]
    

        axis.plot('M:' + mean_stats['M'].astype(str) + '/N:' + mean_stats['N'].astype(str),
            mean_stats['runtime'],
        linestyle="solid",
        color=colors_array[index])

        axis.set_xlabel('M:/N:')
        axis.set_ylabel('runtime (log(seconds))')
        axis.set_yscale('log')

    plt.title(title)
    plt.legend(input_interation_values)
    plt.show()

def plotKeyGraph(df, key_input) :
    for index, input in enumerate(input_interation_values):
        df_plot = [df['optimizationVersion'] == key_input]
        
        mean_stats = df_plot.groupby(['optimizationVersion','nSteps']).mean().reset_index()
        
        axis.plot(mean_stats['nSteps'],
            mean_stats['runtime'],
        linestyle="solid",
        color=colors_array[index])

        #axis.set_xlabel('nbcores (number)')
        #axis.set_ylabel('runtime (log(seconds))')
        #axis.set_yscale('log')


    # Scatter plot
    # for index, input in enumerate(input_interation_values):
    #     df_plot = df[(df['input'] == input)]
    #     df_plot = df_plot[df_plot['version'] == key_input]

    #     axis.scatter(df_plot['nbcore'], 
    #     df_plot['runtime'],
    #     color=colors_array[index])
    #     axis.set_xlabel('nbcores (number)')
    #     axis.set_ylabel('runtime (log(seconds))')
    #     axis.set_yscale('log')
    
    plt.legend(input_interation_values)
    plt.show()

def plotCompareBestOpenMPcuda() :
    bestOpenMP = openMP.groupby(['version','nbcore','M','N']).mean().reset_index()
    bestOpenMP = bestOpenMP[bestOpenMP['version'] == 'parallel']
    bestOpenMP = bestOpenMP[bestOpenMP['nbcore'] == 8].reset_index()
    bestOpenMP['runtime'] = bestOpenMP['runtime'] + 0.0001 # to avoid log(0)

    bestCuda = cuda2.groupby(['optimizationVersion','M','N']).mean().reset_index()
    bestCuda = bestCuda[bestCuda['optimizationVersion'] == 'full-reduction']

    bestOpenMP['M'] = 'M:' + bestOpenMP['M'].astype(str) + '/N:' + bestOpenMP['N'].astype(str)
    bestCuda['M'] = 'M:' + bestCuda['M'].astype(str) + '/N:' + bestCuda['N'].astype(str)

    axis.plot(bestOpenMP['M'],
            bestOpenMP['runtime'],
        linestyle="solid",
        color=colors_array[0])
    
    axis.plot(bestCuda['M'],
            bestCuda['runtime'],
        linestyle="solid",
        color=colors_array[1])

    axis.set_xlabel('M')
    axis.set_ylabel('runtime (log(seconds))')
    axis.set_yscale('log')
    #axis.set_xscale('log')

    plt.legend(['OpenMP', 'Cuda'])
    plt.title("Comparaison des meilleurs temps d'execution")
    plt.show()

#plotCompare("sequential", "parallel", [input_interation_values[-1]])

#plotCompare("simd", "parallel", [input_interation_values[-1]])

#plotGraph(cuda, "Temps d'execution en fonction du nombre de pas")

plotGraph(cuda2, "Temps d'execution en fonction de M et N")

#plotCompareBestOpenMPcuda()