import csv
import numpy as np
from ALP import model_ALP as model
import sys
import os

mass_flag = sys.argv[1]
c_value = sys.argv[2]
beta_value = sys.argv[3]

param_file_dir = "../model_setup/output/"
output_file_dir = "./output/"


file_name = (
    "c" + str(int(c_value)) + "_beta_pi" + str(int(beta_value)) + "_" + str(mass_flag)
)
param_file_name = file_name + "_param.csv"
output_file_name = file_name + "_out.csv"
param_file = os.path.join(param_file_dir, param_file_name)
output_file = os.path.join(output_file_dir, output_file_name)

print(
    "Running for f = "
    + str(int(c_value))
    + " f_c and beta = "
    + str(int(beta_value))
    + "."
)

dataset = []
with open(param_file) as csv_file:
    reader = csv.reader(csv_file)
    for i in reader:
        dataset.append(i)


with open(output_file, "w") as f:
    data_writer = csv.writer(f)
    for point in dataset:
        mS = float(point[0])
        sintheta = float(point[1])
        lh = float(point[2])
        Ap = float(point[3])
        muhsq = float(point[4])
        muSsq = float(point[5])
        f = float(point[6])
        md = model(mS, sintheta, lh, Ap, muhsq, muSsq, f, np.pi / 10)
        print(
            "Scanning mS = "
            + str(md.mS)
            + ", sin theta = "
            + str(md.sintheta)
            + ", lh = "
            + str(md.lh)
            + ", A = "
            + str(md.A)
            + ", muhsq = "
            + str(md.muHsq)
            + ", muSsq = "
            + str(md.muSsq)
            + ", f = "
            + str(md.f)
        )
        md.getTc()
        print("Critical temperature: " + str(md.Tc))
        md.find_Tn()
        strength_1d = md.strength_Tn_1d()
        strength = md.strength_Tn()
        strength_Tc = md.strength_Tc
        data_writer.writerow([mS, sintheta, strength, strength_1d, strength_Tc])
        print("mS = " + str(mS) + ", sin theta = " + str(sintheta) + " scanning done.")
        print("PT strength: " + str(strength))
