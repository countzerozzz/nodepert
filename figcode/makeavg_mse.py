import npimports
from npimports import *

path = "explogs/avg_mseexp/"

expdata = pickle.load(open(path + "delta_mse.pickle", "rb"))

fig, ax1 = pp.subplots()
keys=[]
for kk in expdata.keys():
    keys.append(kk)

test_acc = 100*expdata[keys[0]]['test_acc']
delta_mse = expdata[keys[0]]['delta_mse']

ax1.plot(test_acc, '.-', lw=1)
# ax1.plot(lr, sgddelta_l, lw=1)

# N = len(expdata.keys())
N = 2
pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Reds(np.linspace(0.5,1,N)))

ax2 = ax1.twinx()

# (lr, npdelta_l, sgddelta_l) = expdata[keys[-1]]
ax2.plot(delta_mse, '-', lw=1)
# ax2.plot(lr, sgddelta_l, lw=1)

# N = len(expdata.keys())
# N = 2
# pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Blues(np.linspace(0.5,1,N)))

# ax1.plot(-5.0, 15.0, lw=1)
# ax1.set_xscale('log')
# pp.ylim(-0.4,3.0)

# ax2.plot(-5.0, 15.0, lw=1)
# ax2.set_yscale('log')
# ax2.set_xscale('log')

fig.show()
pp.savefig('figcode/avg_mse.png', dpi=500)
