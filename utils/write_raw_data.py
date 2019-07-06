import os
import csv

raw_data_dir='/mnt/efs/Trees/'
f = open("../raw_data_path.csv", "w")
writer = csv.writer(f)
data_types = ['_RGB-Ir.tif', '_DSM.tif', '_TREE.png']

for subdir in os.listdir(raw_data_dir):
    subdir = os.path.join(raw_data_dir, subdir)
    for data in os.listdir(subdir):
        line = [data + dt for dt in data_types]

        line = [os.path.join(subdir, data, d) for d in line]
        writer.writerow(line)

f.close()


