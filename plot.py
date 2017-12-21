import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

path = "./fastStorage/*.csv"

cpu_usage_all = []
mem_usage_all = []
network_usage_all = []
disk_usage_all = []
sample_rate = 30
num_rows = 8000

color_one = '#3399FF'
color_two = '#FF8000'



for fname in glob.glob(path):

    with open(fname, 'r') as infh:
        next(infh)
        reader = csv.reader(infh, delimiter=';')

        cpu_usage_list = []
        mem_usage_list = []
        disk_usage_list = []
        network_usage_list = []


        for row in reader:
            cpu_usage_list.append(float(row[4]))
            mem_usage_list.append(float(row[6]))
            disk_usage_list.append(abs(float(row[7]) - float(row[8])))
            network_usage_list.append(abs(float(row[9]) - float(row[10])))


        if len(cpu_usage_list) > num_rows:
            cpu_usage_all.append(cpu_usage_list[:num_rows:sample_rate])

        if len(mem_usage_list) > num_rows:
            mem_usage_all.append(mem_usage_list[:num_rows:sample_rate])

        if len(disk_usage_list) > num_rows:
            disk_usage_all.append(disk_usage_list[:num_rows:sample_rate])

        if len(network_usage_list) > num_rows:
            network_usage_all.append(network_usage_list[:num_rows:sample_rate])


cpu_average = np.mean(np.array(cpu_usage_all), axis=0)
mem_average = np.mean(np.array(mem_usage_all), axis=0)
disk_average = np.mean(np.array(disk_usage_all), axis=0)
network_average = np.mean(np.array(network_usage_all), axis=0)


plt.rc('axes', labelsize=16)    # fontsize of the x and y labels

fig, ax1 = plt.subplots()

t = np.arange(0, num_rows, sample_rate)
ax1.plot(t, cpu_average, color=color_one)
ax1.set_xlabel('Timestamp index  #', )
ax1.set_ylabel('CPU average usage (%)', color=color_one)
ax1.tick_params('y', colors=color_one)

ax2 = ax1.twinx()
ax2.plot(t, mem_average, color=color_two)
ax2.set_ylabel('Memory usage (KB)', color=color_two)
ax2.tick_params('y', colors=color_two)

plt.title("Bitbrains CPU and Memory Average Trace", fontsize=24)

plt.show()

plt.rc('axes', labelsize=16)    # fontsize of the x and y labels

fig, ax1 = plt.subplots()

t = np.arange(0, num_rows, sample_rate)
ax1.plot(t, disk_average, color=color_one)
ax1.set_xlabel('Timestamp index  #', )
ax1.set_ylabel('Abs disk usage (KB/s)', color=color_one)
ax1.tick_params('y', colors=color_one)

ax2 = ax1.twinx()
ax2.plot(t, network_average, color=color_two)
ax2.set_ylabel('Abs network usage (KB/s)', color=color_two)
ax2.tick_params('y', colors=color_two)

plt.title("Bitbrains Absolute I/O Average Trace", fontsize=24)

plt.show()






