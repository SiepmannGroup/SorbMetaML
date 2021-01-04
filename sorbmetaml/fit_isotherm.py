import os, argparse
from inspect import signature
import numpy as np
from scipy.optimize import curve_fit

# pre-exponential factor given by statistical mechanics as A = A0/T**2.5
# expression is A = h^3 / ((2*pi*m)**1.5 * (k*T)**2.5)
# where m is the mass of the H2 molecule 
A0 = 13.4662


class Isotherm:
    # temperature is scaled by 1000
    def _arrhenius(a, b, invT):
        return np.exp(a - b * invT)

    def _statmech(a, b, invT):
        return np.exp(a - b * invT + 3.5 * np.log(invT))

    def sips(x, q, a, b, n):
        p, invT = np.exp(x[0]), x[1]
        K = Isotherm._arrhenius(a, b, invT)
        return q * (K * p) ** n / (1 + (K * p) ** n)

    def langmuir(x, q, a, b):
        return Isotherm.sips(x, q, a, b, 1)

    def dslangmuir(x, q1, a1, b1, q2, a2, b2):
        return Isotherm.langmuir(x, q1, a1, b1) + Isotherm.langmuir(x, q2, a2, b2)

    def quadratic(x, q, a1, b1, a2, b2):
        p, invT = np.exp(x[0]), x[1]
        K1 = Isotherm._arrhenius(a1, b1, invT)
        K2 = Isotherm._arrhenius(a2, b2, invT)
        return q * p * (K1 + 2 * K2 * p) / (1 + K1 * p + K2 * p ** 2)

    def sips_s(x, q, a, b, n):
        p, invT = np.exp(x[0]), x[1]
        K = Isotherm._statmech(a, b, invT)
        return q * (K * p) ** n / (1 + (K * p) ** n)

    def langmuir_s(x, q, a, b):
        return Isotherm.sips_s(x, q, a, b, 1)

    def dslangmuir_s(x, q1, a1, b1, q2, a2, b2):
        return Isotherm.langmuir_s(x, q1, a1, b1) + Isotherm.langmuir_s(x, q2, a2, b2)

    def quadratic_s(x, q, a1, b1, a2, b2):
        p, invT = np.exp(x[0]), x[1]
        K1 = Isotherm._statmech(a1, b1, invT)
        K2 = Isotherm._statmech(a2, b2, invT)
        return q * p * (K1 + 2 * K2 * p) / (1 + K1 * p + K2 * p ** 2)

    def __init__(self, name):
        try:
            self._function = getattr(Isotherm, name.lower())
        except AttributeError:
            raise ValueError('Invalid isotherm function: %s' % name.lower())

    def get(self):
        return self._function


def parse():
    parser = argparse.ArgumentParser(description='Fit adsorption data using isotherm functions')
    parser.add_argument('train', type=str, help='Training set to load; should be numpy .npy format')
    parser.add_argument('name', type=str, help='name of adsorption isotherm, langmuir|sips|dslangmuir|quadratic|best')
    parser.add_argument('--test', type=str, default='', help='test set to load; same as training set by default')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    data_raw = np.load(args.train)
    data_test = np.load(args.test) if args.test != '' else data_raw

    if args.name == 'best':
        func = None
    else:
        func = Isotherm(args.name).get()


    mse = []
    popts = []
    for i in range(data_raw.shape[0]):
        x_fit = data_raw[i, 1:, :]
        y_fit = data_raw[i, 0, :]
        x_eval = data_test[i, 1:, :]
        y_true = data_test[i, 0, :]
        try:
            if func:
                nparams = len(signature(func).parameters) - 1
                popt, pcov = curve_fit(func, x_fit, y_fit, bounds=(-50, 50), loss='soft_l1', max_nfev=5000)
                popts.append(popt)
                y_pred = func(x_eval, *popt)
            else:
                func_best = None
                popt_best = None
                mse_best = np.inf
                for n in ['langmuir', 'sips', 'dslangmuir', 'quadratic']:
                    try:
                        fcur = Isotherm(n).get()
                        # for the best isotherm a smaller bound may be used to prevent overfit
                        popt, pcov = curve_fit(fcur, x_fit, y_fit, 
                                               bounds=(-10, 10), loss='soft_l1', max_nfev=5000)
                        y_pred_val = fcur(x_fit, *popt)
                        mse_cur = np.mean((y_fit - y_pred_val) ** 2)
                        if mse_cur < mse_best:
                            mse_best = mse_cur
                            func_best = n
                            popt_best = popt
                    except RuntimeError:
                        print("zeolite %d failed to fit %s isotherm" % (i, n))  
                fcur = Isotherm(func_best).get()
                popts.append(func_best)
                y_pred = fcur(x_eval, *popt_best)
            mse.append(np.mean((y_true - y_pred) ** 2))
            nparams = len(popts)
        except RuntimeError:
            print("zeolite %d failed to fit isotherm" % i)
            popts.append(np.array([np.nan] * nparams))
            mse.append(np.nan)
    print("Average MSE:", np.mean(np.array(mse)))
    if func:
        out = np.append(np.vstack(popts), np.array(mse).reshape(-1, 1), axis=1)
        np.savetxt('%s-fit.csv' % args.name, out, delimiter=',')
    else:
        np.savetxt('%s-fit.csv' % args.name, np.array(mse), delimiter=',')
        np.savetxt('%s-isotherms.csv' % args.name, popts, delimiter=',', fmt='%s')