import sys
import pandas as pd
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python convert_csv.py [path_to_csv_file]")
    exit()

df = pd.read_csv(sys.argv[1])
# group by material
material_names = df['zeolite'].unique()
# normalization factor
norms = []

n_materials = len(material_names)
n_states = len(df.loc[df['zeolite'] == material_names[0]])
np_data = np.zeros((n_materials, 3, n_states))
for i, x in enumerate(material_names):
    rows = df.loc[df['zeolite'] == x]
    norms.append(rows['loading (v)'].max())
    for j in range(len(rows)):
        row = rows.iloc[j]
        np_data[i, :, j] = np.array([
            row["loading (v)"] / norms[i],
            np.log(row["pressure"]),
            1000 / row["temperature"]
        ])

np.savetxt("names.csv", material_names, fmt="%s")
np.savetxt("norms.csv", norms)
np.save(sys.argv[1].split("/")[-1][:-4] + ".npy", np_data)