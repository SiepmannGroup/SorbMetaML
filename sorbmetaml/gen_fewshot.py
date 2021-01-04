import os, argparse
import numpy as np 

np.random.seed(1)

def get_samples(x1, x2, n, mode):
    if mode == 'random':
        return np.random.permutation(x1.shape[0])[:n]
    indices = None
    x1max, = np.where(x1 == np.max(x1))
    x2max, = np.where(x2 == np.max(x2))
    x1min, = np.where(x1 == np.min(x1))
    x2min, = np.where(x2 == np.min(x2))
    corners = [np.intersect1d(x1min, x2min),
               np.intersect1d(x1min, x2max),
               np.intersect1d(x1max, x2min),
               np.intersect1d(x1max, x2max),
              ]
    if mode == 'edge':
        x1half, = np.where(x1 == np.sort(x1)[len(x1)//2])
        x2half, = np.where(x2 == np.sort(x2)[len(x2)//2])
        midpoints = [np.intersect1d(x1min, x2half),
                     np.intersect1d(x1half, x2min),
                     np.intersect1d(x1max, x2half),
                     np.intersect1d(x1half, x2max),
                    ]
        return np.array(corners + midpoints).ravel()
    elif mode == 'diagonal':
        x1diagl, = np.where(x1 == np.sort(x1)[len(x1)//3])
        x1diagh, = np.where(x1 == np.sort(x1)[2*len(x1)//3])
        x2diagl, = np.where(x2 == np.sort(x2)[len(x2)//3])
        x2diagh, = np.where(x2 == np.sort(x2)[2*len(x2)//3])
        midpoints = [np.intersect1d(x1diagl, x2diagl),
                     np.intersect1d(x1diagl, x2diagh),
                     np.intersect1d(x1diagh, x2diagl),
                     np.intersect1d(x1diagh, x2diagh),
                    ]
        return np.array(corners + midpoints).ravel()
    else:
        raise ValueError("Invalid sampling mode!")
        
def parse():
    parser = argparse.ArgumentParser(description='Generate few-shot learning dataset')
    parser.add_argument('path', type=str, help='Path to original dataset in NPY format')
    parser.add_argument('n', type=int, default=8, help='Number of samples in each task')
    parser.add_argument('mode', type=str, default="random", help='Sampling mode, random|edge|diagonal')
    parser.add_argument('--test', action="store_true", default="random", help='Dry run, does not save')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    data_all = np.load(args.path)
    samples = get_samples(data_all[0, 1, :], data_all[0, 2, :], args.n, args.mode)
    print("Samples:")
    print(samples)
    print(list(zip(np.exp(data_all[0, 1, samples]),1000 / data_all[0, 2, samples])))
    data_few = []
    for x in data_all:
        data_few.append(x[:, samples])
    data_few = np.array(data_few)
    if not args.test:
        if args.mode == 'random':
            np.save('%s_%s%d.npy' % (os.path.splitext(args.path)[0], args.mode, args.n), data_few)
        else:
            np.save('%s_%s.npy' % (os.path.splitext(args.path)[0], args.mode), data_few)
