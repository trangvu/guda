import matplotlib.pyplot as plt
import csv

data = {}
header = []

with open('score-cdf.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i == 0:
            header = row[1:]
            continue
        data[row[0]] = list(map(float, row[1:]))

labels = data.keys()
markers = ["+", "v", 's', '|', "."]
colors = ["red", "green", "blue", 'purple', 'y']

fig, ax = plt.subplots()
index = -1

for key, value in data.items():
    index += 1
    ax.plot(header, value, marker=markers[index], color=colors[index], label=key)
ax.set(title='CDF of predicted score')
ax.set(xlabel='Score', ylabel='Cumulative probability')

ax.legend()
fig.tight_layout(pad=1.0)
plt.savefig('cdf.png')