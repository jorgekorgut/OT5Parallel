import os
import subprocess
import csv


try:
    os.remove("stats.csv")
except OSError:
    pass

num_steps = [1e6, 1e8]
nb_core = [1, 2, 4, 6, 8, 12, 24]
repeats = range(0,50)

reduced = {}
divided = {}
atomic = {}


for nbcores in nb_core:
    reduced[nbcores] = {}
    divided[nbcores] = {}
    atomic[nbcores] = {}
    for nsteps in num_steps:
        ncore = 0
        print('Execution of ' + str(nsteps) + ' steps | cores ' + str(nbcores))
        reduced[nbcores][nsteps] = []
        divided[nbcores][nsteps] =[]
        atomic[nbcores][nsteps] = []

        for repeat in repeats:

            args = ("./reduce", "-C", str(nbcores), "-N", str(nsteps))
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()
            reduced[nbcores][nsteps] += [popen.stdout.read()]

            args = ("./divided", "-C", str(nbcores), "-N", str(nsteps))
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()
            divided[nbcores][nsteps] += [popen.stdout.read()]
        #print(divided[nsteps] )

        # args = ("./atomic", "-C", str(ncore), "-N", str(nsteps))
        # popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        # popen.wait()
        # atomic[nsteps] += [popen.stdout.read()]

with open('stats.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     #writer.writerow(['version','nbcore','num_steps','runtime'])
     sumReduced = 0
     for nsteps in num_steps:
        for ncores in nb_core:
            for repeat in repeats:
                writer.writerow(['reduce', ncores, nsteps, reduced[ncores][nsteps][repeat].decode('utf-8')])
                writer.writerow(['divided', ncores, nsteps, divided[ncores][nsteps][repeat].decode('utf-8')])
            #writer.writerow(['atomic', '/', nsteps, atomic[nsteps][repeat]])