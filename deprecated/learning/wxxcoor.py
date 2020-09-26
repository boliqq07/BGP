import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colorbar import ColorbarBase

data = pd.read_excel(r'C:\Users\Administrator\Desktop\wxx\41.xlsx')
cov = data.corr()
cov = cov.fillna(0)
cov.iloc[2, 2] = 1
# cov = pd.DataFrame(corr.cov_shrink)
# cov = cov.set_axis(X_frame_abbr, axis='index', inplace=False)
# cov = cov.set_axis(X_frame_abbr, axis='columns', inplace=False)
fig = plt.figure()
fig.add_subplot(111)

cmap = plt.get_cmap("gnuplot")  # args

sns.heatmap(cov, linewidths=0.3, xticklabels=True, yticklabels=True, square=True, fmt=".1g", vmin=-1, vmax=1, cmap=cmap,
            cbar=False, linecolor="lightgrey",
            annot=True, annot_kws={'size': 5})

ax = plt.axes()
# cbar.set_ylabel(np.linspace(-1, 1, 10))
# cbar.set_ticks(np.linspace(-1, 1, 10))
# cbar.set_ticklabels(('-1', '-0.8', '-0.6', '-0.4', '-0.2', '0', '0.2', '0.4',"0.6","0.8","1"))
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
fig.subplots_adjust(right=0.75)
cbar_ax = fig.add_axes([0.78, 0.125, 0.03, 0.72])
ColorbarBase(cbar_ax, cmap=cmap)
fig.set_size_inches(7, 6, forward=True)
plt.show()
plt.show()
