import npimports
from npimports import *

path = "explogs/fig1exp/"

npexpdata = pickle.load(open(path + "npexpdata.pickle", "rb"))
npparams = pickle.load(open(path + "npparams.pickle", "rb"))
sgdexpdata = pickle.load(open(path + "sgdexpdata.pickle", "rb"))
sgdparams = pickle.load(open(path + "sgdparams.pickle", "rb"))

path = "explogs/fig1exp/"

npexpdata2 = pickle.load(open(path + "npexpdata.pickle", "rb"))
npparams2 = pickle.load(open(path + "npparams.pickle", "rb"))
# sgdexpdata2 = pickle.load(open(path + "sgdexpdata.pickle", "rb"))
# sgdparams2 = pickle.load(open(path + "sgdparams.pickle", "rb"))

npexpdata2.pop(1.25e-4)
npexpdata.pop(1e-5)
# npexpdata.pop(1e-3)
npexpdata.pop(1e-2)
npexpdata.pop(1e-1)
sgdexpdata.pop(1e-5)
# sgdexpdata.pop(1e-6)
sgdexpdata.pop(1e-1)
# sgdexpdata.pop(1e-3)
# sgdexpdata.pop(1e-2)

# merge data:
for (kk,vv) in npexpdata2.items():
    npexpdata.update({kk:vv})
# for (kk,vv) in sgdexpdata2.items():
#     sgdexpdata.update({kk:vv})

print(sorted(npexpdata.keys()))
print(sorted(sgdexpdata.keys()))

fig, ax1 = pp.subplots()

for kk in sorted(npexpdata.keys()):
    epoch = npexpdata[kk]['epoch']
    acc = npexpdata[kk]['test_acc']
    acc = 100.*np.asarray(acc)
    ax1.plot(epoch[1:], acc[1:], '.-', lw=1)

N = len(sgdexpdata.keys())
pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Reds(np.linspace(0.5,1,N)))

ax2 = ax1.twinx()

for kk in sorted(sgdexpdata.keys()):
    epoch = sgdexpdata[kk]['epoch']
    acc = sgdexpdata[kk]['test_acc']
    acc = 100.*np.asarray(acc)
    ax2.plot(epoch[1:], acc[1:], '-', lw=2)


N = len(npexpdata.keys())
pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Blues(np.linspace(0.5,1,N)))


ax1.plot(0.0, 100.0, lw=1)
# ax1.set_yscale('log')
ax1.set_xscale('log')

ax2.plot(0.0, 100.0, lw=1)
# ax2.set_yscale('log')
ax2.set_xscale('log')

fig.show()

# todo: start all the params from the same seed