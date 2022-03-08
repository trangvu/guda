import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.ticker as mtick
data = {
    'law': {},
    'med': {},
    'ted': {}
}
header = []

for domain in ['law', 'ted', 'med']:
    with open("{}.csv".format(domain)) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                header = row[1:]
                continue
            tmp = list(map(float, row[1:]))
            data[domain][row[0]] = [i * 100 for i in tmp]

labels = ["{}-gram".format(n) for n in header]
methods = data['law'].keys()
markers = ["+", "v", 's', '|', "."]
colors = ["green", "blue", 'm', "red", 'c']

for domain in ['law', 'ted', 'med']:
    vals = data[domain]
    X = np.arange(len(labels))
    ind = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots()

    for index, method in enumerate(methods):
        bar = ax.bar(ind + width * index, vals[method], color=colors[index],
                     width=width, label=method,)
        # ax.bar_label(bar, padding=3)

    ax.set_ylabel('Percentage')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Percentage of newly introduced n-gram in {} De-En translation".format(domain))
    ax.set_xticks(ind + width * 2)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout(pad=1)
    plt.savefig("ngram-{}.png".format(domain))

fig, ax = plt.subplots(nrows=1, ncols=3)
for i, domain in enumerate(['law', 'med', 'ted']):
    vals = data[domain]
    X = np.arange(len(labels))
    ind = np.arange(len(labels))
    width = 0.15

    for index, method in enumerate(methods):
        bar = ax[i].bar(ind + width * index, vals[method],
                     width=width, label=method,)
        # ax.bar_label(bar, padding=3)
    ax[i].yaxis.set_major_formatter(mtick.PercentFormatter())
    ax[i].set_title("{} De-En".format(domain))
    ax[i].set_xticks(ind + width * 2)
    ax[i].set_xticklabels(labels)
ax[0].set_ylabel('Percentage')
ax[2].legend()
fig.set_figwidth(12)
fig.set_figheight(3)
fig.tight_layout(pad=1.0)

plt.savefig("ngram-all.png")