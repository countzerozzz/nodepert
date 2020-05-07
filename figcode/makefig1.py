import npimports
from npimports import *

import matplotlib
import matplotlib.pyplot as pp

path = "explogs/fig1exp/"

npexpdata = pickle.load(open(path + "npexpdata.pickle", "rb"))
npparams = pickle.load(open(path + "npparams.pickle", "rb"))

sgdexpdata = pickle.load(open(path + "sgdexpdata.pickle", "rb"))
sgdparams = pickle.load(open(path + "sgdparams.pickle", "rb"))


fig, ax1 = pp.subplots()

N = len(npexpdata.keys())
pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Blues(np.linspace(0,1,N)))
ax1.set_yscale('log')
ax1.set_xscale('log')

for ii, kk in zip(range(len(npexpdata.keys())), npexpdata.keys()):
    epoch = npexpdata[kk]['epoch']
    acc = npexpdata[kk]['test_acc']
    acc = 100*np.asarray(acc)
    # pp.plot(epoch, acc, color=nplinecolors[kk], lw=3)
    ax1.plot(epoch, acc, lw=3)

ax2 = ax1.twinx()
ax2.set_yscale('log')
ax2.set_xscale('log')

N = len(npexpdata.keys())
pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Oranges(np.linspace(0,1,N)))

for ii, kk in zip(range(len(sgdexpdata.keys())), sgdexpdata.keys()):
    epoch = sgdexpdata[kk]['epoch']
    acc = sgdexpdata[kk]['test_acc']
    acc = 100*np.asarray(acc)
    ax2.plot(epoch, acc, lw=3)

fig.show()

# todo:
# run train and test set first before training epoch!
# start from same initial parameters each time!
