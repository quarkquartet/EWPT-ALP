import csv
import numpy as np
from ALP import model_ALP as model

dataset = []
with open("../model_setup/output/C50_beta_pi10_GeV_param.csv") as csv_file:
    reader = csv.reader(csv_file)
    for i in reader:
        dataset.append(i)

output_file = "./output/c50_beta_pi10_GeV_out.csv"

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
