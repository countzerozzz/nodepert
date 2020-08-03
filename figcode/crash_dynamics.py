import npimports
from npimports import *

path = "explogs/crash_dynamics/"
start_time = time.time()

sns.set()
# sns.set_palette(palette="paired")
sns.set_palette(palette="muted")
# sns.set_palette(palette="Paired") #Spectral
# sns.set_palette(palette="Spectral")

grad_norms_df = pd.read_csv(path + 'grad_norms.csv')
graddiff_norms_df = pd.read_csv(path + 'graddiff_norms.csv')
grad_angles_df = pd.read_csv(path + 'grad_angles.csv')
sign_symmetry_df = pd.read_csv(path + 'sign_symmetry.csv')

# df.dropna(axis='index', how='any', inplace=True)
# df.reset_index(inplace=True, drop=True)

pd.set_option('display.max_columns', None)
print(sign_symmetry_df.head(5))

pp.ylim((0, 1)) 

g = sns.FacetGrid(grad_norms_df, col="update_rule")
g.map(sns.lineplot, "batch", "gnorm_w1", marker='.')
pp.savefig('figs/grad_normsw1.png', dpi=500)

g = sns.FacetGrid(grad_norms_df, col="update_rule")
g.map(sns.lineplot, "batch", "gnorm_w3", marker='.')
pp.savefig('figs/grad_normsw3.png', dpi=500)


g = sns.FacetGrid(graddiff_norms_df, col="update_rule")
g.map(sns.lineplot, "batch", "gdiff_norm_w1", marker='.')
pp.savefig('figs/gdiff_norm_w1.png', dpi=500)

g = sns.FacetGrid(graddiff_norms_df, col="update_rule")
g.map(sns.lineplot, "batch", "gdiff_norm_w3", marker='.')
pp.savefig('figs/gdiff_norm_w3.png', dpi=500)


g = sns.FacetGrid(grad_angles_df, col="update_rule")
g.map(sns.lineplot, "batch", "gangle_w1", marker='.')
pp.savefig('figs/gangle_w1.png', dpi=500)

g = sns.FacetGrid(grad_angles_df, col="update_rule")
g.map(sns.lineplot, "batch", "gangle_w3", marker='.')
pp.savefig('figs/gangle_w3.png', dpi=500)


g = sns.FacetGrid(sign_symmetry_df, col="update_rule")
g.map(sns.lineplot, "batch", "ss_w1", marker='.')
pp.savefig('figs/ss_w1.png', dpi=500)

g = sns.FacetGrid(sign_symmetry_df, col="update_rule")
g.map(sns.lineplot, "batch", "ss_w3", marker='.')
pp.savefig('figs/ss_w3.png', dpi=500)
# g.lineplot(x='batch', y='ss_w1', style = 'update_rule', data=sign_symmetry_df, marker='.')
# sns.lineplot(x='batch', y='ss_w3', style = 'update_rule', label='normalize', data=sign_symmetry_df, marker='*')


print(time.time() - start_time)

