from gg_trends import Reg
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from mediacov import search_df
from raw_data_files import time_series

fig, axs = plt.subplots(1, 5, sharey=True, gridspec_kw={'wspace': 0})

fig.set_size_inches(16, 6)

x = y = 0

for issue in Reg:
    train_l = len(time_series) - 5

    selected_series = time_series.copy()

    s_model = SRMA(endog=selected_series[[issue]][:train_l],
                   exog=selected_series[[x for x in selected_series.columns if x != issue]][:train_l],
                   order=(3, 1, 1), seasonal_order=(1, 0, 1, 7)).fit()

    f_ru = selected_series[[issue]].copy()[1:]
    f_ru.columns = ["act"]
    f_ru["pred"] = s_model.predict(end=dt.datetime(2019, 10, 6), endog=selected_series[[issue]][-5:],
                                   exog=selected_series[[x for x in selected_series.columns if x != issue]][-5:],
                                   dynamic=False)

    testing = f_ru.copy()

    testing["err"] = np.abs((testing["act"] - testing["pred"]) / testing["act"])
    fit = round(testing[testing["act"] != 0].error.mean() * 100)
    testing2 = testing[-5:]
    fit_p = round(testing2[testing2["act"] != 0].error.mean() * 100)

    search_df.loc[issue, "topic_model"] = fit
    search_df.loc[issue, "topic_predicted"] = fit_p

    f_ru["act"].plot(title="{}\nMAPE: test: {}% model: {}%".format(issue, fit_p, fit), ax=axs[x])
    f_ru["pred"][:-5].plot(color="orange", label="predicted: Train", ax=axs[x])
    f_ru["pred"][-6:].plot(color="red", label="predicted: Test", ax=axs[x])

    if x == 0:
        handles, labels = axs[0].get_legend_handles_labels()

    x += 1

fig.subplots_adjust(right=.2)
fig.tight_layout()
lgd = fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.05, .5))

axs[0].set_ylabel("% Search or # Articles")
fig.suptitle("Model and Predicted Google Search Results", y=1.05)
