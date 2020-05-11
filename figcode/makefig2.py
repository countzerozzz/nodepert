import npimports
from npimports import *

path = "explogs/fig2exp/"

expdata = pickle.load(open(path + "linesearch_data.pickle", "rb"))

fig, ax1 = pp.subplots()
keys=[]
for kk in expdata.keys():
    keys.append(kk)

(lr, npdelta_l, sgddelta_l) = expdata[keys[-1]]
ax1.plot(lr, npdelta_l, '.-', lw=1)
ax1.plot(lr, sgddelta_l, lw=1)

# N = len(expdata.keys())
N = 2
pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Reds(np.linspace(0.5,1,N)))

# ax2 = ax1.twinx()

# (lr, npdelta_l, sgddelta_l) = expdata[keys[-1]]
# ax2.plot(lr, npdelta_l, '.-', lw=1)
# ax2.plot(lr, sgddelta_l, lw=1)

# N = len(expdata.keys())
# N = 2
# pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Blues(np.linspace(0.5,1,N)))

# ax1.plot(-5.0, 15.0, lw=1)
ax1.set_xscale('log')
pp.ylim(-0.4,3.0)

# ax2.plot(-5.0, 15.0, lw=1)
# ax2.set_yscale('log')
# ax2.set_xscale('log')

fig.show()
pp.savefig('figcode/fig2.png', dpi=500)
