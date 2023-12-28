import os
import subprocess
import csv


try:
    os.remove("Analysis/Cuda2/stats.csv")
except OSError:
    pass

values_M = [2, 4, 8, 10, 12, 14] #16 is not allowed
values_N = [1, 3, 7, 9, 11, 13]
values_index_range = range(0, len(values_N)) 

repeats = range(0,10)
results = {}
results["naif"] = []
results["reduction"] = []
results["full-reduction"] = []

for k,v in results.items():
    for index_N in values_index_range:
        results[k].append([]) 

for value_index in values_index_range:
    print('M:'+str(values_M[value_index])+ '/N:' + str(values_N[value_index]))

    for repeat in repeats:
        args = ("./Executables/cuda_vector.o", "-M", str(values_M[value_index]),"-N", str(values_N[value_index]))
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        popen.wait()
        results["naif"][value_index] += [popen.stdout.read()]

        args = ("./Executables/cuda_vector_reduction.o", "-M", str(values_M[value_index]), "-N", str(values_N[value_index]))
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        popen.wait()
        results["reduction"][value_index] += [popen.stdout.read()]

        args = ("./Executables/cuda_vector_full-reduction.o", "-M", str(values_M[value_index]),"-N", str(values_N[value_index]))
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        popen.wait()
        results["full-reduction"][value_index] += [popen.stdout.read()]
            


with open('Analysis/Cuda2/stats.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     for k,v in results.items():
        for value_index in values_index_range:
            for repeat in repeats:
                writer.writerow([k,values_M[value_index], values_N[value_index], results[k][value_index][repeat].decode('utf-8')])