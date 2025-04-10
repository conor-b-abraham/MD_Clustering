import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys

def roundup(num):
    roundnum = round(num, 2)
    if roundnum-0.01 < num:
        roundnum += 0.01
    return roundnum

sils = np.loadtxt('silhouette_scores.dat')

plt.rcParams["font.size"] = 8
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams['axes.linewidth'] = 2

ymax = roundup(np.max(sils[:,1]))

fig, ax = plt.subplots(figsize=(3.5,2), tight_layout=True, dpi=300)

if len(sys.argv) == 2:
    ax.axvline(int(sys.argv[1]), color="k", linestyle="solid", linewidth=0.75)
else:
    ax.vlines(x=sils[np.argmax(sils[:,1]),0], ymin=0, ymax=ymax, linestyles='dotted', colors='k', linewidth=0.75)
# annoopts = {'rotation':'vertical', 'ha':'center', 'va':'center', 'backgroundcolor':'white'}
# ax.annotate(f'{int(sils[np.argmax(sils[:,1]),0])} Clusters', xy = (sils[np.argmax(sils[:,1]),0], np.max(sils[:,1])/2), **annoopts)
ax.plot(sils[:,0], sils[:,1], color='tab:red')#, marker='.')
ax.set_xlabel('# of Clusters')
ax.set_ylabel('Silhouette Score')
ax.set_xlim(0, 100)#np.max(sils[:,0]))
ax.set_ylim(0, ymax)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
# ax.tick_params(which='both', width=2)
ax.grid(linestyle="dotted")
plt.savefig('mpl_silhouette_scores.png', transparent=True)
plt.show()
