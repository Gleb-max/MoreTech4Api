from raw_data_files import time_series
import datetime as dt

fig, axs = plt.subplots(4, 5, sharex=True)

fig.set_size_inches(16, 12)

x = y = 0

for issue in time_series:

    if not issue.find(".com") > -1:
        continue

    train_l = len(time_series) - 5

    selected_series = time_series[[col for col in time_series.columns if (col.find(issue[:issue.find("_")]) > -1)]]

    s_model = SRMA(endog=selected_series[[issue]][:train_l][1:],
                   exog=selected_series[[x for x in selected_series.columns if x != issue]][
                        :train_l].shift().add_suffix("_l1")[1:],
                   order=(3, 1, 1), seasonal_order=(1, 0, 1, 7)).fit()

    f_ru = selected_series[[issue]].copy()[1:]
    f_ru.columns = ["act"]
    f_ru["pred"] = s_model.predict(end=dt.datetime(2019, 10, 6), endog=selected_series[[issue]][-5:],
                                        exog=selected_series[
                                                 [x for x in selected_series.columns if x != issue]].shift()[-5:],
                                        dynamic=False)

    testing = f_ru.copy()

    testing["err"] = np.abs((testing["act"] - testing["pred"]) / testing["act"])
    fit = round(testing[testing["act"] != 0].error.mean() * 100)

    mape_df.loc[issue, "WithinNewspapersOnly_model"] = fit

    testing2 = testing[-5:]
    fit_p = round(testing2[testing2["act"] != 0].error.mean() * 100)
    mape_df.loc[issue, "WithinNewspapersOnly_predicted"] = fit_p

    f_ru["act"].plot(title="{}\nMAPE: test: {}% model: {}%".format(issue, fit_p, fit), ax=axs[x, y])
    f_ru["pred"][:-5].plot(color="orange", label="predicted: Train", ax=axs[x, y])
    f_ru["pred"][-6:].plot(color="red", label="predicted: Test", ax=axs[x, y])

    x += 1
    if x > 3:
        x = 0
        y += 1

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc=' right')

fig.suptitle("News hin publication")
