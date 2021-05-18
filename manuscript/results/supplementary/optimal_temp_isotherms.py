import numpy as np
import pandas as pd
import sys
from scipy.optimize import curve_fit


material = "iza" if len(sys.argv) == 1 else sys.argv[1]
root = "../%s" % material

data_all = np.load("%s/%s_hydrogen.npy" % (root, material))
zeolites = np.genfromtxt("%s/names.csv" % root, dtype=str)
norms = np.loadtxt("%s/norms.csv" % root)


R = 0.08314
eos = lambda p, V, T: p * V / R / T # does not work > 50 bar

def sips(x, q, a, b, n):
    p, invT = np.exp(x[0]), x[1]
    K = np.exp(a - b * invT)
    return q * (K * p) ** n / (1 + (K * p) ** n)

def langmuir(x, q, a, b):
    return sips(x, q, a, b, 1)

def dslangmuir(x, q1, a1, b1, q2, a2, b2):
    return langmuir(x, q1, a1, b1) + langmuir(x, q2, a2, b2)

def quadratic(x, q, a1, b1, a2, b2):
    p, invT = np.exp(x[0]), x[1]
    K1 = np.exp(a1 - b1 * invT)
    K2 = np.exp(a2 - b2 * invT)
    return q * p * (K1 + 2 * K2 * p) / (1 + K1 * p + K2 * p ** 2)

def get_ypred_isotherm(func, x, data_train):
    y_all = []
    y_train = data_train[:, 0]
    x_train = data_train[:, 1:].T
    try:
        popt, pcov = curve_fit(func, x_train, y_train, bounds=(-50, 50), loss='soft_l1', max_nfev=5000)
    except RuntimeError:
        popt, pcov = curve_fit(func, x_train, y_train, bounds=(-50, 50), loss='soft_l1', max_nfev=100000)
    return func(x.T, *popt)

def get_optimal_temps(func, p0, p1, e):
    rows = []
    for i in range(data_all.shape[0]):
        data = data_all[i].T
        temps = np.arange(45, 279, 0.1)
        size = len(temps)
        xs_0 = np.zeros((size, 2), dtype=np.float32)
        xs_1 = np.zeros((size, 2), dtype=np.float32)
        xs_0[:, 1] = 1000 / temps
        xs_1[:, 1] = 1000 / temps
        xs_0[:, 0] = np.log(p0)
        xs_1[:, 0] = np.log(p1)
        n_zeo = 2.0158 * get_ypred_isotherm(func, np.vstack([xs_0, xs_1]), data) * norms[i]
        q0 = n_zeo[:size]
        q1 = n_zeo[size:]
        n_work = (q1 - q0) * e + ((eos(p1, 2.0158, temps) - eos(p0, 2.0158, temps)) * (1 - e))
        # first search maximum > 77 K, otherwise extrapolate
        tid = np.argmax(n_work[temps >= 77.0])
        if tid == 0:
            tid = np.argmax(n_work)
        else:
            tid += np.where(abs(temps - 77.0) < 0.01)[0][0]
        t_opt = temps[tid]
        rows.append([t_opt, n_work[tid]])
    return np.array(rows)


if __name__ == "__main__":
    p0 = 2.71
    p1 = 30
    e = 0.7
    funcs = [langmuir, sips, quadratic, dslangmuir]
    func_names = ["Langmuir", "Sips", "Quadratic", "DSL"]
    cols = ["Zeolite"]
    for name in ["Meta-learning"] + func_names:
        cols += ["%s T_max" % name, "%s Delta_n" % name]
    rows = []
    data_ml = pd.read_csv("%s/temps-p0%s-p%s-f%.1f-swing0.csv" % (root, p0, p1, e))
    rows.append(zeolites[..., None])
    rows.append(data_ml["Optimal temperature (K)"].values[..., None])
    rows.append(data_ml["Tank capacity (mol)"].values[..., None])
    for func in funcs:                                     
        rows.append(get_optimal_temps(func, p0, p1, e))
    rows = np.hstack(rows)
    print(rows.shape)
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("optimal_temps_comparison_%s.csv" % material)
    