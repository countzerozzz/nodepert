import npimports
from npimports import *

path = "explogs/adamupdate/"
start_time = time.time()

# expdata1 = pickle.load(open(path + "normalize.pkl", "rb"))
# expdata2 = pickle.load(open(path + "standardize.pkl", "rb"))
# expdata3 = pickle.load(open(path + "zca.pkl", "rb"))

sns.set()
# sns.set_palette(palette="paired")
sns.set_palette(palette="muted")
# sns.set_palette(palette="Paired") #Spectral
# sns.set_palette(palette="Spectral")
# fig, ax1 = pp.subplots()

# epoch = expdata1['epoch']
# acc = expdata1['test_acc']
# acc = np.asarray(acc)
# ax1.plot(epoch, acc, '.-', lw=1)

# epoch = expdata2['epoch']
# acc = expdata2['test_acc']
# acc = np.asarray(acc)
# ax1.plot(epoch, acc, '.-', lw=1)
# df=pd.read_csv('myfile.csv', sep=',',header=None)

df = pd.DataFrame(columns = ['test_acc', 'apply_grad', 'update_rule'])

for update_rule in ['np', 'sgd']:
    for apply_grad in ['vanilla', 'adam']:
        tmp = pd.DataFrame()
        tmp['test_acc'] = np.genfromtxt(path+update_rule+apply_grad+'.csv', delimiter=',', dtype='float')
        tmp['apply_grad'] = pd.Series((tmp['test_acc'].size)*[apply_grad])
        tmp['update_rule'] = pd.Series(25*[update_rule.upper()])
        df = df.append(tmp, ignore_index=True)        

df.dropna(axis='index', how='any', inplace=True)
df.reset_index(inplace=True, drop=True)

pd.set_option('display.max_columns', None)
print(df.head(10))

pp.ylim((0, 100)) 

sns.violinplot(x='update_rule', y='test_acc', hue='apply_grad', data=df, inner='points', scale='count')
pp.savefig('figs/npadam-violin.png', dpi=500)

pp.clf()
sns.stripplot(x='update_rule', y='test_acc', hue='apply_grad', data=df)
pp.savefig('figs/npadam.png', dpi=500)

print(time.time() - start_time)


# pp.ylabel("test_acc")
# df = pd.DataFrame({'X_Axis':[1,3,5,7,10,20],
#                    'col_2':[.4,.5,.4,.5,.5,.4],
#                    'col_3':[.7,.8,.9,.4,.2,.3],
#                    'col_4':[.1,.3,.5,.7,.1,.0],
#                    'col_5':[.5,.3,.6,.9,.2,.4]})

# df = df.melt('X_Axis', var_name='cols',  value_name='vals')
# g = sns.catplot(x="X_Axis", y="vals", hue='cols', data=df)

# df['epoch'] = np.arange(start=1,stop=31)
# df['normalize'] = 100*np.array([0.3767, 0.4048, 0.4653, 0.4927, 0.5166, 0.5354, 0.5326, 0.5297, 0.545, 0.559, 0.5718, 0.5795, 0.5839, 0.5766, 0.6004, 0.5941, 0.597, 0.5951, 0.6192, 0.6188, 0.6125, 0.6208, 0.6194, 0.6239, 0.62, 0.6103, 0.6233, 0.6231, 0.6266, 0.6216])
# df['standardize'] = 100*np.array([0.4584999978542328, 0.5236999988555908, 0.4945000112056732, 0.5533000230789185, 0.578499972820282, 0.5911999940872192, 0.6097999811172485, 0.6284000277519226, 0.6230000257492065, 0.6352999806404114, 0.645799994468689, 0.6333000063896179, 0.6517000198364258, 0.6482999920845032, 0.642799973487854, 0.641700029373169, 0.6565999984741211, 0.6532999873161316, 0.6333000063896179, 0.646399974822998, 0.6531000137329102, 0.6485999822616577, 0.6481999754905701, 0.6353999972343445, 0.6438000202178955, 0.6363000273704529, 0.6450999975204468, 0.6319000124931335, 0.6288999915122986, 0.6205999851226807])
# df['zca'] = 100*np.array([0.1279, 0.1188, 0.1674, 0.279, 0.304, 0.3315, 0.3895, 0.4428, 0.4379, 0.5119, 0.5208, 0.5234, 0.5216, 0.55, 0.537, 0.5627, 0.5671, 0.5652, 0.5903, 0.5848, 0.5699, 0.5837, 0.5968, 0.6075, 0.6122, 0.5426, 0.6091, 0.6129, 0.5954, 0.609])

# sns.lineplot(x='epoch', y='normalize', label='normalize', data=df, marker='*')
# sns.lineplot(x='epoch', y='standardize', label='standardize', data=df, marker='o')
# sns.lineplot(x='epoch', y='zca', label='zca', data=df, marker='<')

# for kk in sorted(expdata3.keys()):
#     epoch = expdata3[kk]['epoch']
#     acc = expdata3[kk]['test_acc']
#     acc = 100.*np.asarray(acc)
#     ax1.plot(epoch, acc, '.-', lw=1)

# ax1.plot(0.0, 100.0, lw=1)
# ax1.set_yscale('log')
# ax1.set_xscale('log')

