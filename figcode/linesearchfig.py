import matplotlib.pyplot as pp
import matplotlib
import seaborn as sns
import pandas as pd
import pickle 
import time

start_time = time.time()
path = "explogs/analysis/linesearch/"

expdata = pickle.load(open(path + "expdata.pkl", "rb"))

sns.set()
keys=[]
for kk in expdata.keys():
    print(kk)
    keys.append(kk)

df = pd.DataFrame(expdata[keys[1]])
df['sgd'] = df['sgd'].astype('float32')
df['np'] = df['np'].astype('float32')

print(df.head())
print(df.columns)
print(df.info())

fig, ax1 = pp.subplots()
pp.ylim(-0.5,1.0)
ax1.set_xscale('log')
sns.lineplot(x='lr', y='sgd', err_style="bars", markers=True, data=df, palette="muted")
sns.lineplot(x='lr', y='np', markers=True, data=df, palette="muted")

pp.savefig('figcode/linesearch.png', dpi=500)

print(time.time() - start_time)

# keys=[]
# for kk in expdata.keys():
#     keys.append(kk)

# (lr, npdelta_l, sgddelta_l) = expdata[keys[-1]]
# ax1.plot(lr, npdelta_l, '.-', lw=1)
# ax1.plot(lr, sgddelta_l, lw=1)

# # N = len(expdata.keys())
# N = 2
# pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Reds(np.linspace(0.5,1,N)))

# # ax2 = ax1.twinx()

# # (lr, npdelta_l, sgddelta_l) = expdata[keys[-1]]
# # ax2.plot(lr, npdelta_l, '.-', lw=1)
# # ax2.plot(lr, sgddelta_l, lw=1)

# # N = len(expdata.keys())
# # N = 2
# # pp.rcParams["axes.prop_cycle"] = pp.cycler("color", pp.cm.Blues(np.linspace(0.5,1,N)))

# # ax1.plot(-5.0, 15.0, lw=1)
# ax1.set_xscale('log')
# pp.ylim(-0.4,3.0)

# # ax2.plot(-5.0, 15.0, lw=1)
# # ax2.set_yscale('log')
# # ax2.set_xscale('log')

# fig.show()
# pp.savefig('figcode/fig2.png', dpi=500)
