import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('logs/new_test.csv', sep='|', header=None)

df_ttts = df[df[0] == 'tm_ttts']
df_unif = df[df[0] == 'tm_uniform']
df_ts = df[df[0] == 'tm_ts']

ttts_allocations = df_ttts.groupby(1).mean()
unif_allocations = df_unif.groupby(1).mean()
ts_allocations = df_ts.groupby(1).mean()

optimal_allocation = [
    4.297370300122487635e-03,
    7.240575883226209146e-03,
    1.431574287114295574e-02,
    4.014697113038132731e-02,
    4.339119070387972288e-01,
    4.441608521332328641e-01,
    3.711800774201511055e-02,
    1.255197537221579805e-02,
    6.089414414605934245e-03]

fig, ax = plt.subplots()

ax.plot(range(0, 9), ttts_allocations.iloc[0], label='txts 250')
# ax.plot(range(0, 9), ttts_allocations.iloc[1], label='txts 500')
# ax.plot(range(0, 9), ttts_allocations.iloc[2], label='txts 750')
ax.plot(range(0, 9), ttts_allocations.iloc[3], label='txts 1000')
# ax.plot(range(0, 9), ttts_allocations.iloc[4], label='txts 1250')
# ax.plot(range(0, 9), ttts_allocations.iloc[5], label='txts 1500')
# ax.plot(range(0, 9), ttts_allocations.iloc[6], label='txts 1750')
ax.plot(range(0, 9), ttts_allocations.iloc[7], label='txts 2000')
ax.plot(range(0, 9), np.ones(9) / 9, label='uniform')
ax.plot(range(0, 9), ts_allocations.iloc[0], label='ts 250')
# ax.plot(range(0, 9), ts_allocations.iloc[1], label='ts 500')
# ax.plot(range(0, 9), ts_allocations.iloc[2], label='ts 750')
ax.plot(range(0, 9), ts_allocations.iloc[3], label='ts 1000')
# ax.plot(range(0, 9), ts_allocations.iloc[4], label='ts 1250')
# ax.plot(range(0, 9), ts_allocations.iloc[5], label='ts 1500')
# ax.plot(range(0, 9), ts_allocations.iloc[6], label='ts 1750')
ax.plot(range(0, 9), ts_allocations.iloc[7], label='ts 2000')
ax.plot(range(0, 9), optimal_allocation, label='fixed constrained optimal')
plt.legend()
plt.show()


# a = np.loadtxt(fname='logs/arms_argmin_10000.csv')
#
# df = pd.DataFrame(a)
#
# means = df.mean(axis=0)
#
# print(means[5:9].sum())


allocations =  [i for i in ttts_allocations.iloc[7]]
# plt.bar(range(0, 9), allocations)
# plt.axvline(x=5)
# plt.show()

bottom = 0

fig, ax = plt.subplots()
for i in range(9):
    plt.bar(0, allocations[i], bottom=bottom, width=.1)
    bottom += allocations[i]
ax.axhline(y=.5)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.show()
