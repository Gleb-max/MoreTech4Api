import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from raw_data_files import time_series

Reg = ["dprk", "mkw", "rss"]

fig, axs = plt.subplots(1, 5, sharey=True, gridspec_kw={'wspace': 0})

fig.set_facecolor("white")
fig.set_size_inches(24, 6)
idx = 0

for rg in Reg:
    tmp = time_series[[x for x in time_series.columns if (x.find(rg) > -1)]]
    tmp.rename(columns={rg: (rg + " search")}).plot(
        style=[":" if not x.startswith(rg) else "-" for x in tmp], title="{}".format(rg), ylim=(0, 100),
        ax=axs[idx])
    idx += 1

axs[0].set_ylabel("Selection/Day")

fig.suptitle("Google Cov", y=1)
