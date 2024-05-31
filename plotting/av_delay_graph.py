import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from plotting.av_delay_results import model_a_results, model_b_results, model_c_results

plt.style.use('seaborn-v0_8')
plt.rc('axes', axisbelow=True)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.5


models = {
    model_a_results['name']: model_a_results,
    model_b_results['name']: model_b_results,
    model_c_results['name']: model_c_results,
}

true_delay = model_a_results['true_delay']

all_model_types = []
all_predicted_delays = []
all_probabilities = []

for model, results in models.items():
    all_model_types.extend([model] * len(results["predictions"]))
    all_predicted_delays.extend(results["predictions"])
    all_probabilities.extend(results["probabilities"])

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.axhline(true_delay, c='green', lw=1, ls='dashed', zorder=1)
plt.scatter(all_model_types, all_predicted_delays, s=100, alpha=all_probabilities)

plt.axhline(0, c='black', linewidth=0.5, zorder=0)
plt.grid(linewidth=0.25, c='darkgrey')
plt.yticks(np.arange(-2, 2.2, step=0.2))
plt.xlabel('Model Type')
plt.ylabel('Predicted Delay (s)')
plt.title('Predicted Delays with Probabilities')

plt.show()
