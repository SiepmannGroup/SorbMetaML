import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c 

def z_eos(x, a22, a21, a12, a20, a02, a11, a10, a01):
    p, inv_t = np.exp(x[0]), x[1]
    return a22 * p**2 * inv_t**2 + a21 * p**2 * inv_t + a12 * p * inv_t**2 + a20 * p ** 2\
            + a02 * inv_t ** 2 + a11 * p * inv_t + a10 * p + a01 * inv_t

def fit_z_pressure(z_data, pressure):
    z_true = z_data[z_data[:, 0] == np.log(pressure), 2]
    temps = z_data[z_data[:, 0] == np.log(pressure), 1]
    popt, pcov = curve_fit(quadratic, temps, z_true)
    z_pred = quadratic(temps, *popt)
    return popt, z_true, z_pred

def fit_z_all(z_data):
    z_true = z_data[:, 2]
    popt, pcov = curve_fit(z_eos, z_data[:, :2].T, z_true - 1)
    z_pred = z_eos(z_data[:, :2].T, *popt) + 1
    return popt, z_true, z_pred

r_sq = lambda x, y: np.corrcoef(x, y)[0,1] ** 2

# columns: log(p), 1000/T, Z
z_data = np.loadtxt('compressibility.csv', delimiter=',')
z_data[:, 1] /= 1000
pressures = [1.0, 2.71, 7.39, 20.09, 30.0, 54.60, 148.4, 403.4]
popts = []
z_fit = []
lines = []
for p in pressures:
    popt, z_true, z_pred = fit_z_pressure(z_data, p)
    popts.append(np.log(np.abs(popt)))
    lines.append([p] + popt.tolist() + [r_sq(z_true, z_pred)])
    z_fit.append(z_true)
    z_fit.append(z_pred)
#    plt.scatter(z_true, z_pred)
#plt.show()
#popts = np.array(popts)
#plt.scatter(np.log(pressures), popts[:, 0])
#plt.scatter(np.log(pressures), popts[:, 1])
#plt.show()

#popt, z_true, z_pred = fit_z_all(z_data)
#print(r_sq(z_true, z_pred))
#plt.scatter(z_true, z_pred)
#plt.show()

print(" & ".join(["%.4g" % (l[1]*0.01) for l in lines]), "\\\\")
print(" & ".join(["%.4g" % (l[2]) for l in lines]), "\\\\")
print(" & ".join(["%.4g" % (l[3]) for l in lines]), "\\\\")
print(" & ".join(["%.4g" % (l[4]) for l in lines]), "\\\\")
np.savetxt('z_fit.txt', np.vstack(z_fit).T)
