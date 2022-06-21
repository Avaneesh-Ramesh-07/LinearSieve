# Test the ability to recover clusters of weakly correlated variables
# The weak clusters (relatively low TC) are often noisy.
# It seems that Bayesian smoothing has a big effect on this noise.
# Here we test various
import sys
sys.path.append( '..' )


import numpy as np
import LinearSieve.vis_sieve as vs
from scipy.stats import kendalltau
import LinearSieve.linearsieve as linearsieve


verbose = False
np.set_printoptions(precision=3, suppress=True, linewidth=200)
seed = 1
np.random.seed(seed)
colors = ['black', 'red', 'green', 'blue', 'yellow', 'indigo', 'gold', 'hotpink', 'firebrick', 'indianred',
          'mistyrose', 'darkolivegreen', 'darkseagreen', 'pink', 'tomato', 'lightcoral', 'orangered',
          'palegreen', 'darkslategrey', 'greenyellow', 'burlywood', 'seashell', 'mediumspringgreen',
          'papayawhip', 'blanchedalmond', 'chartreuse', 'dimgray', 'peachpuff', 'springgreen', 'aquamarine',
          'orange', 'lightsalmon', 'darkslategray', 'brown', 'ivory', 'dodgerblue', 'peru', 'darkgrey',
          'lawngreen', 'chocolate', 'crimson', 'forestgreen', 'slateblue', 'cyan', 'mintcream', 'silver']

# 30 groups, 5-20 variables each, various weak correlations
# Some groups bimodal.
# Trying to mimic qualitative features of rnaseq and ADNI dataset
n_samples = 500
n_groups = 30


def standardize(s):
    return (s - np.mean(s)) / np.std(s)


def get_r(s1, s2):
    return np.mean(standardize(s1) * standardize(s2))


def observed(s):
    # Generate a randomly sized group of variables weakly correlated to
    bimodal_s = s + 0.5 * (s > 0.2).astype(float)  # occasional unbalanced, bimodal
    n = np.random.randint(3, 16)
    ns = len(s)

    output = []
    for i in range(n):
        noise_mag = np.random.choice([0.2, 0.4, 0.6])
        signal_mag = np.random.choice([0.1, 0.5, 1])
        if np.random.random() < 0.05:
            this_s = bimodal_s
        else:
            this_s = s
        output.append(signal_mag * this_s + noise_mag * np.random.randn(ns))
    return np.vstack(output).T


def score(true, predicted):
    """Compare n true signals to some number of predicted signals.
    For each true signal take the min RMSE of each predicted.
    Signals are standardized first."""
    rs = []
    for t in true.T:
        rs.append(max(np.abs(kendalltau(t, p)[0]) for p in predicted.T))
    return np.array(rs)


baseline = np.random.random((n_samples, n_groups))
signal = np.random.random((n_samples, n_groups))
signal = (signal - np.mean(signal, axis=0, keepdims=True)) / np.std(signal, axis=0, keepdims=True)
data_groups = [observed(s) for s in signal.T]
order = np.argsort([-q.shape[1] for q in data_groups])
data_groups = [data_groups[i] for i in order]
signal = np.array([signal[:,i] for i in order]).T
data = np.hstack(data_groups)
print(('group sizes', [q.shape[1] for q in data_groups]))
print(('Data size:', data.shape))

for loop_i in range(1):
    out = linearsieve.Sieve(n_hidden=n_groups, seed=seed + loop_i, verbose=verbose).fit(data)
    print ('Done, scoring:')
    scores = score(signal, out.transform(data))
    print(('TC:', out.tc))
    print(('Actual score:', scores))
    print(('Number Ok, %d / %d' % (np.sum(scores > 0.5), len(scores))))
    print(('total score, %0.3f' % np.sum(scores)))

names = []
for j, group in enumerate(data_groups):
    color = colors[j]
    rs = [get_r(q, signal[:, j]) for q in group.T]
    mis = [(-0.5 * np.log(1 - r**2)) for r in rs]
    #print ','.join(map(lambda r: '%0.3f' % r, mis))
    #print(list(np.sum(mis)))
    #np.sum(mis) is of type map
    print((('Color: %s\tNumber in group: %d\ttotal MI: %0.3f' % (color, group.shape[1], list(np.sum(mis)))).expandtabs(30)))
    for i in range(group.shape[1]):
        names.append(color + '_' + str(i))

vs.vis_rep(out, data, column_label=names, prefix='weak')
print(('Perfect score:', score(signal, signal)))
print(('Baseline score:', score(signal, baseline)))