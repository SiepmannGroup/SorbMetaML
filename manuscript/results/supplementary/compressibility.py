import numpy as np
from scipy.optimize import curve_fit

def quadratic(x, a0, a1, a2):
    return a2 * x ** 2 + a1 * x + a0 

def fit_z_pressure(z_data, pressure):
    z_true = z_data[z_data[:, 0] == np.log(pressure), 2]
    temps = z_data[z_data[:, 0] == np.log(pressure), 1]
    popt, pcov = curve_fit(quadratic, temps, z_true)
    z_pred = quadratic(temps, *popt)
    return popt, z_true, z_pred


r_sq = lambda x, y: np.corrcoef(x, y)[0,1] ** 2

# original columns: log(p), 1000/T, Z
z_data = np.loadtxt('z_data.csv', delimiter=',')
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
    
# print compressibility factors
for t in np.sort(np.unique(z_data[:, 1]))[::-1]:
    line = "%.2f" % (1 / t)
    

    for p in pressures:
        z = z_data[(z_data[:, 1] == t) & (z_data[:, 0] == np.log(p)), 2][0]
        if z > 1.05:
            line += " & \\textbf{%.3f}" % z
        else:
            line += " & %.3f" % z
    line += "\\\\"
    print(line)
        
# print quadratic fits
print(" & ".join(["%.4g" % (l[1]) for l in lines]), "\\\\")
print(" & ".join(["%.4g" % (l[2]) for l in lines]), "\\\\")
print(" & ".join(["%.4g" % (l[3]*0.01) for l in lines]), "\\\\")
print(" & ".join(["%.4g" % (l[4]) for l in lines]), "\\\\")
np.savetxt('z_fit.txt', np.vstack(z_fit).T)