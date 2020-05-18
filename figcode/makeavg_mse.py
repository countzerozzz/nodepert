import npimports
from npimports import *
import seaborn as sns
import pandas as pd

path = "explogs/avg_mseexp/"

expdata = pickle.load(open(path + "delta_mse5e_4.pickle", "rb"))
sns.set_style("dark")
# sns.set()

fig, ax1 = pp.subplots()
ax2 = ax1.twinx()

num=500

x=np.arange(0, num, 1)

# N = len(expdata.keys())
# 'b','g','','r'
colors=["light navy", "light forest green", "terra cotta", "deep violet"]

c1=sns.xkcd_rgb[colors[0]]
c2=sns.xkcd_rgb[colors[1]]
c3=sns.xkcd_rgb[colors[2]]
c4=sns.xkcd_rgb[colors[3]]

ax1.plot(x, [y * 100 for y in expdata[1]['test_acc'][:num]], '-', lw=1.5, color=c1, alpha=1, label='1hl')
ax1.plot(x, [y * 100 for y in expdata[2]['test_acc'][:num]], '-', lw=1.5, color=c2, alpha=1, label='2hl')
ax1.plot(x, [y * 100 for y in expdata[3]['test_acc'][:num]], '-', lw=1.5, color=c3, alpha=1, label='3hl')
ax1.plot(x, [y * 100 for y in expdata[4]['test_acc'][:num]], '-', lw=1.5, color=c4, alpha=1, label='4hl')

ax1.set_ylabel('test acc')
ax1.set_yticks([50, 100])

# ax1.plot(x, test_acc[1000:1100], '-', lw=1)
# ax1.plot(x, test_acc[1500:1600], '-', lw=1)
# ax1.plot(lr, sgddelta_l, lw=1)

# N=2
# pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Blues(np.linspace(0.5,1,N)))


# (lr, npdelta_l, sgddelta_l) = expdata[keys[-1]]

# N = len(expdata.keys())

x_half=np.arange(0, num, 0.5)

# delta_mse1 = pd.rolling_mean(delta_mse[:2*num],5)

def get_sma(mylist, N):
    return np.convolve(mylist, np.ones((N,))/N, mode='same')

cnt=0

ax2.yaxis.grid(color='gray', linestyle='dashed')
ax1.set_axisbelow(True)

ax2.plot(x_half, get_sma(expdata[1]['delta_mse'][:num*2], 5), '-', lw=1, color=c1, alpha=0.6)
ax2.plot(x_half, get_sma(expdata[2]['delta_mse'][:num*2], 5), '-', lw=1, color=c2, alpha=0.6)
ax2.plot(x_half, get_sma(expdata[3]['delta_mse'][:num*2], 5), '-', lw=1, color=c3, alpha=0.6)
ax2.plot(x_half, get_sma(expdata[4]['delta_mse'][:num*2], 5), '-', lw=1, color=c4, alpha=0.6)
ax1.set_xlabel('Epochs')
ax2.set_ylabel('average MSE')

ax1.legend(loc='best')

# N=1
# pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Reds(np.linspace(0.5,1,N)))

ax2.set_ylim(bottom=-0.002, top=0.002)


# ax2.set_yscale('log')


# for kk in expdata.keys():
#     x_half=np.arange(0, len(test_acc), 0.5)
#     delta_mse = expdata[kk]['delta_mse']
#     ax2.plot(x_half[:2*num], delta_mse[:2*num], '-', lw=1)
#     ax2.set_ylim(bottom=-0.01, top=0.01)
# ax2.plot(lr, sgddelta_l, lw=1)


# ax1.plot(-5.0, 15.0, lw=1)
# pp.ylim(-0.4,3.0)

# ax2.plot(-5.0, 15.0, lw=1)
# ax2.set_yscale('log')
# ax2.set_xscale('log')

fig.show()
pp.tight_layout()
pp.savefig('figcode/avg_mse.png', dpi=600)
