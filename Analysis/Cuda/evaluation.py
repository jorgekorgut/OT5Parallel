import os
import subprocess
import csv


try:
    os.remove("Analysis/Cuda/stats.csv")
except OSError:
    pass

values_N = [1e4, 1e5, 1e6, 1e7, 1e8]
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
    ncore = 0
    print('Execution of N:' + str(values_N[value_index]))

    for repeat in repeats:
        args = ("./Executables/cuda_pi.o","-N", str(values_N[value_index]))
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        popen.wait()
        results["naif"][value_index] += [popen.stdout.read()]

        args = ("./Executables/cuda_pi_reduction.o", "-N", str(values_N[value_index]))
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        popen.wait()
        results["reduction"][value_index] += [popen.stdout.read()]

        args = ("./Executables/cuda_pi_full_reduction.o", "-N", str(values_N[value_index]))
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        popen.wait()
        results["full-reduction"][value_index] += [popen.stdout.read()]
            


with open('Analysis/Cuda/stats.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     for k,v in results.items():
        for value_index in values_index_range:
            for repeat in repeats:
                writer.writerow([k, 'N:'+str(values_N[value_index]), results[k][value_index][repeat].decode('utf-8')])