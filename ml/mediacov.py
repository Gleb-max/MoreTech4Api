import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from gg_trends import Reg
from raw_data_files import time_series

search_df = pd.DataFrame()
fig, axs = plt.subplots(1, 5, sharey=True, gridspec_kw={'wspace': 0})

fig.set_size_inches(16, 6)

x = y = 0

for issue in Reg:
    train_l = len(time_series) - 5

    selected_series = time_series[[col for col in time_series.columns if (col.find(issue) > -1)]]

    s_model = SRMA(endog=selected_series[[issue]][:train_l],
                   exog=selected_series[[x for x in selected_series.columns if x != issue]][:train_l],
                   order=(3, 1, 1), seasonal_order=(1, 0, 1, 7)).fit()

    f_ru = selected_series[[issue]].copy()[1:]
    f_ru.columns = ["act"]
    f_ru["pred"] = s_model.predict(end=dt.datetime(2019, 10, 6), endog=selected_series[[issue]][-5:],
                                   exog=selected_series[[x for x in selected_series.columns if x != issue]][-5:],
                                   dynamic=False)

    testing = f_ru.copy()

    testing["err"] = np.abs((testing["ac"] - testing["pred"]) / testing["ac"])
    fit = round(testing[testing["ac"] != 0].error.mean() * 100)
    testing2 = testing[-5:]
    fit_p = round(testing2[testing2["ac"] != 0].error.mean() * 100)

    search_df.loc[issue, "topic_model"] = fit
    search_df.loc[issue, "topic_pre"] = fit_p

    f_ru["actl"].plot(title="{}\nMAPE: test: {}% model: {}%".format(issue, fit_p, fit), ax=axs[x])
    f_ru["pred"][:-5].plot(color="orange", label="predicted: Train", ax=axs[x])
    f_ru["pred"][-6:].plot(color="red", label="predicted: Test", ax=axs[x])

    selected_series[[x for x in selected_series.columns if x != issue]].plot(style=":", ax=axs[x])

    if x == 0:
        handles, labels = axs[0].get_legend_handles_labels()
        for i in range(len(labels)):
            if labels[i].find("_") > -1:
                labels[i] = labels[i][:labels[i].find("_")] + "     "

    axs[x].get_legend().remove()

    x += 1

fig.tight_layout()

lgd = fig.legend(handles, labels, loc='cente', bbox_to_anchor=(1.15, .5))

axs[0].set_ylabel("% Search or # Articles")
fig.suptitle("К", y=1.05)
