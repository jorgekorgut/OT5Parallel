import os
import subprocess
import csv


try:
    os.remove("Analysis/Part2/stats.csv")
except OSError:
    pass

values_M = [2, 4, 8, 10, 12, 14, 15] #16 is not allowed
values_N = [1, 3, 7, 9, 11, 13, 15]
values_index_range = range(0, len(values_M)) 

nb_core = [1, 2, 4, 6, 8, 12, 24]
repeats = range(0,10)

sequential = {}
parallel = {}
simd = {}

for nbcores in nb_core:
    sequential[nbcores] = {}
    parallel[nbcores] = {}
    simd[nbcores] = {}
    for value_index in values_index_range:
        ncore = 0
        print('Execution of N:' + str(values_N[value_index]) +'M:' +str(values_M[value_index])+ ' | cores ' + str(nbcores))
        sequential[nbcores][value_index] = []
        parallel[nbcores][value_index] = []
        simd[nbcores][value_index] = []

        for repeat in repeats:
            args = ("./Executables/vector_seq.o", "-C", str(nbcores), "-N", str(values_N[value_index]), "-M", str(values_M[value_index]))
            popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            popen.wait()
            sequential[nbcores][value_index] += [popen.stdout.read()]

            args = ("./Executables/vector_parallel.o", "-C", str(nbcores), "-N", str(values_N[value_index]), "-M", str(values_M[value_index]))
            popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            popen.wait()
            parallel[nbcores][value_index] += [popen.stdout.read()]

            args = ("./Executables/vector_simd.o", "-C", str(nbcores), "-N", str(values_N[value_index]), "-M", str(values_M[value_index]))
            popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            popen.wait()
            simd[nbcores][value_index] += [popen.stdout.read()]
            


with open('Analysis/Part2/stats.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     #writer.writerow(['version','nbcore','num_steps','runtime'])
     sumReduced = 0
     for value_index in values_index_range:
        for ncores in nb_core:
            for repeat in repeats:
                writer.writerow(['sequential', ncores, 'M:'+str(values_M[value_index])+'/N:'+str(values_N[value_index]), sequential[ncores][value_index][repeat].decode('utf-8')])
                writer.writerow(['parallel', ncores, 'M:'+str(values_M[value_index])+'/N:'+str(values_N[value_index]), parallel[ncores][value_index][repeat].decode('utf-8')])
                writer.writerow(['simd', ncores, 'M:'+str(values_M[value_index])+'/N:'+str(values_N[value_index]), simd[ncores][value_index][repeat].decode('utf-8')])