import os
import sys
import json
import numpy as np
import pandas as pd
import cmasher as cmr
from datetime import datetime

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

true_offset = 0.0
plot_mean_pred = True
plot_mode_pred = True
time_indexed_files = True
likelihood_threshold = 0.6
filter_prediction_range = lambda pred_and_prob: (-1 <= pred_and_prob[0]) and (pred_and_prob[0] <= 1)

# input_file = "../output/av-predictions/Graham-Norton.json"
input_file = sys.argv[1]

# Plot global video detection results over all clips in timeline
with open(input_file, 'r') as fp:
    video_detection_results = json.load(fp)

plot_width = 12 + len(video_detection_results.keys()) // 2
point_size = 800

x_axis_vals = []
x_axis_labels = []
y_axis = []
colour_by_prob = []
confident_predictions = []
weighted_prediction_total = 0
weights_total = 0

fig, ax = plt.subplots(1, 1, figsize=(plot_width, 9))

for video_index, (video_id, prediction) in enumerate(video_detection_results.items()):
    # Collate video (x), prediction (y), and likelihood (c) for plotting
    prediction = list(filter(filter_prediction_range, prediction))

    if time_indexed_files:
        times = (
            datetime.strptime(video_id.split('_')[1], '%H:%M:%S.%f'),
            datetime.strptime(video_id.split('_')[2], '%H:%M:%S.%f')
        )

        x_value = f"     {datetime.strftime(times[0], '%H:%M:%S')} \n-> {datetime.strftime(times[1], '%H:%M:%S')}"
    else:
        x_value = video_id

    probs = []
    for pred, prob in prediction:
        x_axis_vals.append(video_index)
        x_axis_labels.append(x_value)
        y_axis.append(pred)
        probs.append(prob)

    colour_by_prob.extend(probs)

    # Plot ring around maximal prediction
    if len(probs) > 0:
        max_likelihood_idx = np.argmax(probs)
        max_likelihood_prediction, max_likelihood = prediction[max_likelihood_idx]
        max_likelihood_prediction = float(max_likelihood_prediction)
        video_index = float(video_index)

        if max_likelihood > likelihood_threshold:
            confident_predictions.append(max_likelihood_prediction)

            weighted_prediction_total += max_likelihood * max_likelihood_prediction
            weights_total += max_likelihood

            if video_index == len(video_detection_results) - 1:
                ax.scatter(video_index, max_likelihood_prediction, s=point_size, facecolors='none', edgecolors='k', linewidth=2, zorder=11, label='Max prediction')
            else:
                ax.scatter(video_index, max_likelihood_prediction, s=point_size, facecolors='none', edgecolors='k', linewidth=2, zorder=11)

# Plot all predictions by likelihood
colour_map = cmr.get_sub_cmap('Greens', start=np.min(colour_by_prob), stop=np.max(colour_by_prob))
predictions_plot = ax.scatter(x_axis_vals, y_axis, c=colour_by_prob, cmap=colour_map, s=point_size, zorder=10)

# Most common and mean offset prediction markers
mode_prediction = max(confident_predictions, key=confident_predictions.count)
weighted_mean_prediction = weighted_prediction_total / weights_total

if plot_mode_pred and plot_mean_pred and round(mode_prediction, 2) == round(weighted_mean_prediction, 2):
    plt.axhline(y=mode_prediction, linestyle='-', c='darkred', linewidth=4, label=f'Mean & mode prediction ({mode_prediction:.2f})')
else:
    if plot_mode_pred:
        plt.axhline(y=mode_prediction, linestyle='-', c='darkred', linewidth=4, label=f'Mode prediction ({mode_prediction:.2f})')

    if plot_mean_pred:
        plt.axhline(y=weighted_mean_prediction, linestyle='-', c='orange', linewidth=4, label=f'Mean prediction ({weighted_mean_prediction:.2f})')

# True offset value marker
if true_offset is not None:
    if plot_mean_pred and (round(weighted_mean_prediction, 2) == round(true_offset, 2) or round(mode_prediction, 2) == round(true_offset, 2)):
        plt.axhline(y=true_offset, linestyle='--', c='steelblue', linewidth=4, label=f'True offset ({true_offset:.2f})')
    else:
        plt.axhline(y=true_offset, linestyle='-', c='steelblue', linewidth=4, label=f'True offset ({true_offset:.2f})')

plt.xticks(fontsize='large', rotation=90)
ax.set_xticks(x_axis_vals)
ax.set_xticklabels(x_axis_labels)
ax.xaxis.set_label_coords(0.5, -0.2)

y_limit = round(round(np.max(np.absolute(y_axis)) / 0.2) * 0.2 + 0.2, 1)
ax.set_yticks(np.arange(-y_limit + 0.2, y_limit, 0.2))
plt.yticks(fontsize='x-large')

ax.set_xlabel("Video Segment Index", fontsize='xx-large')
ax.set_ylabel("Predicted Offset (s)", fontsize='xx-large')

if true_offset is None:
    ax.set_title(f"Predicted AV Offset per Video Segment\n", fontsize=20)
elif true_offset == 0:
    ax.set_title(f"Predicted AV Offset per Video Segment (in sync test clip)\n", fontsize=20)
elif true_offset < 0:
    ax.set_title(f"Predicted AV Offset per Video Segment ({true_offset}s offset test clip)\n", fontsize=20)
elif true_offset > 0:
    ax.set_title(f"Predicted AV Offset per Video Segment (+{true_offset}s offset test clip)\n", fontsize=20)

cbar = fig.colorbar(predictions_plot, ax=ax, orientation='vertical', extend='both', ticks=np.arange(0, 1.1, 0.1), fraction=0.03, pad=0.01)
cbar.set_label(label='Likelihood', fontsize='xx-large')
cbar.ax.tick_params(labelsize='x-large')

plt.legend(loc=2, frameon=True, markerscale=0.5, borderpad=0.7, facecolor='w', fontsize='large').set_zorder(12)
ax.grid(which='major', linewidth=1, zorder=0)
plt.tight_layout()

output_path = os.path.splitext(input_file)[0] + "-plot.png"
print(f" * Predictions plot generated: {output_path}")
plt.savefig(output_path)
plt.close()

no_correct_preds = len([p for p in confident_predictions if p == true_offset])
print(f" * Accuracy on confident predictions: {no_correct_preds} / {len(confident_predictions)}")
print(f" * Accuracy on all segments: {no_correct_preds} / {len(np.unique(x_axis_labels))}")
