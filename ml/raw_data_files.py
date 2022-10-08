import gdelt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gd = gdelt.gdelt(version=1)

import os

os.makedirs("data", exist_ok=True)

import datetime

cur_date = datetime.datetime(2019, 10, 7) - datetime.timedelta(days=60)
end_date = datetime.datetime(2019, 10, 7)

while cur_date < end_date:

    print("%s-%s-%s" % (cur_date.year, cur_date.month, cur_date.day))
    if not os.path.exists("data/%s-%s-%s.pkl" % (cur_date.year, cur_date.month, cur_date.day)):

        year = cur_date.year
        month = str(cur_date.month)
        day = str(cur_date.day)

        if cur_date.month < 10:
            month = "0" + month
        if cur_date.day < 10:
            day = "0" + day

        results = gd.Search(['%s %s %s' % (year, month, day)], table='gkg', coverage=True, translation=False)
        results.to_pickle("data/%s-%s-%s.pkl" % (cur_date.year, cur_date.month, cur_date.day))

    cur_date += datetime.timedelta(days=1)

    df = pd.DataFrame()
k = os.listdir("data")
for i in k:
    print(i)
    if i.endswith(".pkl"):
        tmp = pd.read_pickle("data/" + i)
        tmp = tmp[tmp["SOURCES"].apply(lambda x: x in mySources)]
        df = pd.concat([df, tmp])
df.DATE = df.DATE.apply(lambda x: str(x))
df.DATE = pd.to_datetime(df.DATE)
df.fillna("", inplace=True)
df.set_index("DATE", drop=True, inplace=True)

df["dprk"] = df["LOCATIONS"].apply(lambda x: x.find("dprk") > -1)
df["mkw"] = df["LOCATIONS"].apply(lambda x: x.find("mkw") > -1)
df["rss"] = df["LOCATIONS"].apply(lambda x: x.find("rss") > -1)

loc_df = df.groupby(["SOURCES", "DATE"])[["dprk", "mkw", "rss"]].sum()

mySources = ('www.vedomosti.ru/info/rss',)

time_series = pd.DataFrame()
for publisher in mySources:
    time_series = pd.concat([time_series, loc_df.ix[publisher].add_prefix("{}_".format(publisher))])

    mape_df = pd.DataFrame()

    fig, axs = plt.subplots(4, 5, sharex=True)

    fig.set_size_inches(16, 12)

    x = y = 0

    for issue in time_series:
        if issue.find(".com") < 0:
            continue

    train_l = len(time_series) - 5

    selected_series = time_series[[col for col in time_series.columns if (col.find(issue[issue.find("_"):]) > -1)]]
    pub_series = time_series[[col for col in time_series.columns if (col.find(issue[:issue.find("_")]) > -1)]].drop(
        columns=issue)
    selected_series = selected_series.join(pub_series)

    s_model = SRMA(endog=selected_series[[issue]][:train_l][1:],
                   exog=selected_series[[x for x in selected_series.columns if x != issue]][
                        :train_l].shift().add_suffix("_l1")[1:],
                   order=(3, 1, 1), seasonal_order=(1, 0, 1, 7)).fit()

    f_ru = selected_series[[issue]].copy()[1:]
    f_ru.columns = ["act"]
    f_ru["pred"] = s_model.predict(end=datetime.datetime(2019, 10, 6), endog=selected_series[[issue]][-5:],
                                   exog=selected_series[
                                            [x for x in selected_series.columns if x != issue]].shift()[-5:],
                                   dynamic=False)

    testing = f_ru.copy()

    testing["err"] = np.abs((testing["act"] - testing["pred"]) / testing["act"])
    fit = round(testing[testing["act"] != 0].error.mean() * 100)

    mape_df.loc[issue, "NewspapersOnly_model"] = fit

    testing2 = testing[-5:]
    fit_p = round(testing2[testing2["act"] != 0].error.mean() * 100)
    mape_df.loc[issue, "NewspapersOnly_predicted"] = fit_p

    f_ru["act"].plot(title="{}\nMAPE: test: {}% model: {}%".format(issue, fit_p, fit), ax=axs[x, y])
    f_ru["pred"][:-5].plot(color="orange", label="predicted: Train", ax=axs[x, y])
    f_ru["pred"][-6:].plot(color="red", label="predicted: Test", ax=axs[x, y])

    x += 1
    if x > 3:
        x = 0
        y += 1

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc=' right')

fig.suptitle("external topic")
