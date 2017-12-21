import glob
import csv
import os

path = "./fastStorage/*.csv"
output_path = "./output/"

sample_size_average = 30 # 30 entries = 9000 ms = 9 second average

for fname in glob.glob(path):
    with open(fname, 'r') as infh:
        next(infh)
        reader = csv.reader(infh, delimiter=';')

        output = open(output_path + fname[fname[2:].find('/')+3:], "w+")

        timestamp_list = []
        cpu_usage_list = []
        mem_usage_percent_list = []
        disk_read_list = []
        disk_write_list = []
        net_in_list = []
        net_out_list = []

        counter = 0

        for row in reader:

            timestamp = int(row[0])
            cpu_usage = float(row[4])
            mem_capacity = float(row[5])
            mem_usage = float(row[6])
            disk_read = float(row[7])
            disk_write = float(row[8])
            net_in = float(row[9])
            net_out = float(row[10])

            # Append to lists
            timestamp_list.append(timestamp)
            cpu_usage_list.append(cpu_usage)

            if (mem_capacity != 0):
                mem_usage_percent_list.append((mem_usage/mem_capacity)*100.0)
            else:
                mem_usage_percent_list.append(0.0)

            disk_read_list.append(disk_read)
            disk_write_list.append(disk_write)
            net_in_list.append(net_in)
            net_out_list.append(net_out)

            counter+= 1

            if counter >= sample_size_average:
                # Get the averages
                timestamp_avg = sum(timestamp_list)/len(timestamp_list)
                cpu_avg = sum(cpu_usage_list)/len(cpu_usage_list)
                mem_avg = sum(mem_usage_percent_list)/len(mem_usage_percent_list)
                disk_read_avg = sum(disk_read_list)/len(disk_read_list)
                disk_write_avg = sum(disk_write_list)/len(disk_write_list)
                net_in_avg = sum(net_in_list)/len(net_in_list)
                net_out_avg = sum(net_out_list)/len(net_out_list)

                class_num = -1

                if cpu_avg < 0.25:
                    class_num = 0
                elif cpu_avg < 0.5:
                    class_num = 1
                elif cpu_avg < 1.0:
                    class_num = 2
                elif cpu_avg < 2.0:
                    class_num = 3
                elif cpu_avg < 4.0:
                    class_num = 4
                elif cpu_avg < 8.0:
                    class_num = 5
                elif cpu_avg < 16.0:
                    class_num = 6
                elif cpu_avg < 32.0:
                    class_num = 7
                elif cpu_avg < 64.0:
                    class_num = 8
                else:
                    class_num = 9

                counter = 0
                timestamp_list = []
                cpu_usage_list = []
                mem_usage_percent_list = []
                disk_read_list = []
                disk_write_list = []
                net_in_list = []
                net_out_list = []

                output.write(str(timestamp_avg) + ';' + str(cpu_avg) + ';' + str(mem_avg) + ';' + str(disk_read_avg) + ';' + str(disk_write_avg) + ';' + str(net_in_avg) + ';' + str(net_out_avg) + ';' + str(class_num) + '\n')

