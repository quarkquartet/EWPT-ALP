import csv
import sys

import numpy as np
from ALP import model_ALP as model

param_file = sys.argv[1]
output_file = sys.argv[2]

Tc_flag = sys.argv[3]
pi_factor = float(sys.argv[4])

print("Parameter file: " + param_file)
print("Output file: " + output_file)
print("delta: " + str(pi_factor) + " * pi.")

dataset = []
with open(param_file) as csv_file:
    reader = csv.reader(csv_file)
    for i in reader:
        dataset.append(i)
if Tc_flag == "Tc":
    print("Only search for Tc.")
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
            if len(point) > 7:
                Vdiff = float(point[7])
                eEDM = float(point[8])
                nEDM = float(point[9])
                if Vdiff <= 0:
                    continue
            md = model(mS, sintheta, lh, Ap, muhsq, muSsq, f, np.pi * pi_factor)
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
            Tc = md.Tc
            strength_Tc = md.strength_Tc
            data_writer.writerow(
                [
                    mS,
                    sintheta,
                    md.lh,
                    md.A,
                    md.muHsq,
                    md.muSsq,
                    md.f,
                    strength_Tc,
                    Tc,
                    eEDM,
                    nEDM,
                ]
            )
            print(
                "mS = " + str(mS) + ", sin theta = " + str(sintheta) + " scanning done."
            )
            print("PT strength: " + str(strength_Tc))

else:
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
            md = model(mS, sintheta, lh, Ap, muhsq, muSsq, f, np.pi / 5)
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
            try:
                md.find_Tn()
                strength_1d = md.strength_Tn_1d()
                strength = md.strength_Tn()
                strength_Tc = md.strength_Tc
                Tc = md.Tc
                Tnuc = md.Tn
                Tnuc_1d = md.Tn1d
                beta_H = md.beta_over_H()
                beta_H_1d = md.beta_over_H_1d()
                data_writer.writerow(
                    [
                        mS,
                        sintheta,
                        md.lh,
                        md.A,
                        md.muHsq,
                        md.muSsq,
                        md.f,
                        strength_Tc,
                        Tc,
                        strength,
                        strength_1d,
                        Tnuc,
                        Tnuc_1d,
                        beta_H,
                        beta_H_1d,
                    ]
                )
                print(
                    "mS = "
                    + str(mS)
                    + ", sin theta = "
                    + str(sintheta)
                    + " scanning done."
                )
                print("PT strength: " + str(strength))
            except:
                Tc = md.Tc
                strength_Tc = md.strength_Tc
                data_writer.writerow(
                    [
                        mS,
                        sintheta,
                        md.lh,
                        md.A,
                        md.muHsq,
                        md.muSsq,
                        md.f,
                        strength_Tc,
                        Tc,
                    ]
                )
                continue
