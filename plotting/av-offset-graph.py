import pandas as pd
import matplotlib.pyplot as plt
import math

plt.style.use('seaborn-v0_8')
plt.rc('axes', axisbelow=True)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.5


vocalist = pd.read_csv("./vocalist-predictions.csv")
# vocalist['likelihood'] = vocalist['likelihood'].apply(lambda l: (l - vocalist['likelihood'].min()) / (vocalist['likelihood'].max() - vocalist['likelihood'].min()))
vocalist['likelihood'] = vocalist['likelihood'].apply(lambda l: 2 * l)
likelihood_alpha = vocalist['likelihood'].to_list()

ax = vocalist.plot.scatter(
    x='true-offset',
    y='predicted-offset',
    c='true-offset',
    s=likelihood_alpha,
    colormap='autumn'
)

ax.set_xticks(vocalist['true-offset'])
plt.show()
