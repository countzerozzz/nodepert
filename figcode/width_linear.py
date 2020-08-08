import npimports
from npimports import *

path = "explogs/scaling/"
start_time = time.time()

sns.set()
# sns.set_palette(palette="paired")
sns.set_palette(palette="muted")
# sns.set_palette(palette="Paired") #Spectral
# sns.set_palette(palette="Spectral")


width_linear_df = pd.read_csv(path + 'width_linear.csv')
# grad_angles_df.read_csv(path + 'grad_angles.csv')
# train_df.read_csv(path + 'train_df.csv')

# df.dropna(axis='index', how='any', inplace=True)
# df.reset_index(inplace=True, drop=True)

pd.set_option('display.max_columns', None)

pp.ylim((0, 1)) 

g = sns.FacetGrid(width_linear_df, col="hl_size")
g.map(sns.lineplot, "epoch", "test_acc", marker='.')
pp.savefig('figs/scale_wlin.png', dpi=500)

# g.lineplot(x='batch', y='ss_w1', style = 'update_rule', data=sign_symmetry_df, marker='.')
# sns.lineplot(x='batch', y='ss_w3', style = 'update_rule', label='normalize', data=sign_symmetry_df, marker='*')

# pp.savefig('figs/ss.png', dpi=500)

print(time.time() - start_time)

