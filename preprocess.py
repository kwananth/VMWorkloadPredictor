import glob
import csv
import os

path = "./output/*.csv"
output_path = "./ml_data/"

features_multiplier = 10 # 10 * 7 = 70 features

for fname in glob.glob(path):
    with open(fname, 'r') as infh:
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

        csv_file_list = []

        for row in reader:
            timestamp_avg = float(row[0])
            cpu_avg = float(row[1])
            mem_avg = float(row[2])
            disk_read_avg = float(row[3])
            disk_write_avg = float(row[4])
            net_in_avg = float(row[5])
            net_out_avg = float(row[6])
            class_num = float(row[7])

            '''
            csv_file_list.append(
                str(timestamp_avg) + ';' + str(cpu_avg) + ';' + str(mem_avg) + ';' + str(disk_read_avg) + ';' + str(
                    disk_write_avg) + ';' + str(net_in_avg) + ';' + str(net_out_avg))
            '''

            entry_list = [timestamp_avg, cpu_avg, mem_avg, disk_read_avg, disk_write_avg, net_in_avg, net_out_avg, class_num]
            csv_file_list.append(entry_list)

        print(len(csv_file_list))

        i = 0
        while i < len(csv_file_list) - features_multiplier:
            upperbound = i + features_multiplier

            j = i
            while j < upperbound:
                output.write(str(csv_file_list[j][0]) + ';' + str(csv_file_list[j][1]) + ';' + str(csv_file_list[j][2]) + ';' + str(csv_file_list[j][3]) + ';' + str(
                    csv_file_list[j][4]) + ';' + str(csv_file_list[j][5]) + ';' + str(csv_file_list[j][6]) + ';')
                j += 1
            output.write(str(csv_file_list[j][7]) + '\n')

            i += 1
