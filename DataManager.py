import pandas as pd
import numpy as np
import utils

from collections import defaultdict

Features_Raw = ["Open", "High", "Low", "Close", "Volume", "Price"]
Features_Raw2 = ["open_ratio", "high_ratio", "low_ratio", "close_ratio", "volume_ratio", "Price"]


def get_data(path,
             train_date_start=None, train_date_end=None,
             test_date_start=None, test_date_end=None):
    data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})
    data = data.replace(0, np.nan)
    data = data.fillna(method="bfill")
    data.insert(data.shape[1], "Price", data["Close"].values)
    # data = get_scaling(data)
    data = get_ratio(data)
    data = data.dropna()

    train_date_start = data["Date"].iloc[0] if train_date_start is None else train_date_start
    train_date_end = data["Date"].iloc[-1] if train_date_end is None else train_date_end
    test_date_start = data["Date"].iloc[0] if test_date_start is None else test_date_start
    test_date_end = data["Date"].iloc[-1] if test_date_end is None else test_date_end

    train_data = data[(data["Date"] >= train_date_start) & (data["Date"] <= train_date_end)]
    test_data = data[(data["Date"] >= test_date_start) & (data["Date"] <= test_date_end)]

    train_data = train_data.set_index("Date", drop=True)
    test_data = test_data.set_index("Date", drop=True)

    train_data = train_data.astype("float32")
    test_data = test_data.astype("float32")
    return train_data.loc[:,Features_Raw2], test_data.loc[:,Features_Raw2]


def get_data_tensor(path_list,
                    train_date_start=None, train_date_end=None,
                    test_date_start=None, test_date_end=None):

    for path in path_list:
        train_data, test_data = get_data(path,
                                         train_date_start, train_date_end,
                                         test_date_start, test_date_end)

        if path == path_list[0]:
            common_date_train = set(train_data.index.unique())
            common_date_test = set(test_data.index.unique())
        else:
            common_date_train = common_date_train & set(train_data.index.unique())
            common_date_test = common_date_test & set(test_data.index.unique())

    common_date_train = list(common_date_train)
    common_date_test = list(common_date_test)

    common_date_train.sort()
    common_date_test.sort()

    for path in path_list:
        train_data_, test_data_ = get_data(path,
                                         train_date_start, train_date_end,
                                         test_date_start, test_date_end)

        train_data_ = train_data_[train_data_.index.isin(common_date_train)].to_numpy()
        test_data_ = test_data_[test_data_.index.isin(common_date_test)].to_numpy()

        train_data_ = train_data_[:, :, np.newaxis]
        test_data_ = test_data_[:, :, np.newaxis]

        if path == path_list[0]:
            train_data = train_data_
            test_data = test_data_
        else:
            train_data = np.concatenate([train_data, train_data_], axis=-1)
            test_data = np.concatenate([test_data, test_data_], axis=-1)

    print("-------------------------------------")
    print(f"학습 데이터 시작 거래일:{common_date_train[0]}")
    print(f"학습 데이터 마지막 거래일:{common_date_train[-1]}")
    print(f"테스트 데이터 시작 거래일:{common_date_test[0]}")
    print(f"테스트 데이터 마지막 거래일:{common_date_test[-1]}")
    print("-------------------------------------")
    return train_data, test_data


def get_scaling(data):
    feature_names = list(data.columns)
    not_scaling = ["MACD", "MACDsignal", "MACDoscillator", "Price", "Date"]
    for name in not_scaling:
        if name in data.columns:
            feature_names.remove(name)
    for name in feature_names:
        feature_data = data.loc[:, name]
        feature_mean = feature_data.mean()
        feature_std = feature_data.std()
        data.loc[:, name] = (data.loc[:, name] - feature_mean) / feature_std
    return data


def get_ratio(data):
    day = 1
    data.loc[day:, "open_ratio"] = (data["Open"][day:].values - data["Open"][:-day].values) / data["Open"][:-day].values
    data.loc[day:, "high_ratio"] = (data["High"][day:].values - data["High"][:-day].values) / data["High"][:-day].values
    data.loc[day:, "low_ratio"] = (data["Low"][day:].values - data["Low"][:-day].values) / data["Low"][:-day].values
    data.loc[day:, "close_ratio"] = (data["Close"][day:].values - data["Close"][:-day].values) / data["Close"][:-day].values
    data.loc[day:, "volume_ratio"] = (data["Volume"][day:].values - data["Volume"][:-day].values) / data["Volume"][:-day].values
    return data


def get_covariance():
    path_list = utils.local_path
    return_data = defaultdict(float)
    names = [path.split("/")[-1] for path in path_list]

    for i, path in enumerate(path_list):
        data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})
        data = data.replace(0, np.nan)
        data = data.fillna(method="bfill")

        data = data[data["Date"] <= "2021-12-31"]
        data = data.set_index("Date", drop=True)

        data = data.astype("float32")
        prices = data["Close"].values
        returns = [0] * len(prices)

        for k in range(len(prices)):
            if k <= 29:
                returns[k] = 0.
            else:
                returns[k] = (prices[k] - prices[k - 30]) / prices[k - 30]

        return_data[names[i]] = returns

    len_list = [len(data) for data in return_data.values()]
    min_len = min(len_list)

    for name in names:
        return_data[name] = return_data[name][:min_len]

    return_data = pd.DataFrame(return_data).loc[30:]
    cov = return_data.cov()
    return cov


def get_mean():
    path_list = utils.local_path
    price_data = defaultdict(float)
    names = [path.split("/")[-1] for path in path_list]

    for i, path in enumerate(path_list):
        data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})
        data = data.replace(0, np.nan)
        data = data.fillna(method="bfill")

        data = data[data["Date"] <= "2021-12-31"]
        data = data.set_index("Date", drop=True)

        data = data.astype("float32")
        prices = data["Close"].values
        price_data[names[i]] = prices

    len_list = [len(data) for data in price_data.values()]
    min_len = min(len_list)

    for name in names:
        price_data[name] = price_data[name][:min_len]

    price_data = pd.DataFrame(price_data)
    mean_data = price_data.mean()
    return mean_data


def VaR(stock_list, weight):
    cov = pd.read_csv(utils.DATA_DIR + "/COV", index_col=0)
    cov = cov.loc[stock_list, stock_list]
    weight = np.array(weight)
    variance = weight.T.dot(cov).dot(weight)
    deviation = np.sqrt(variance)
    VaR = 1.65 * 15000000 * deviation
    return VaR


def expected(stock_list, weight):
    mean_data = pd.read_csv(utils.DATA_DIR + "/MEAN", index_col=0)
    mean_data = mean_data.loc[stock_list]
    mean_data = np.array(mean_data).reshape(-1)
    weight = np.array(weight).reshape(-1)
    expected = np.dot(mean_data, weight)
    return expected


def variance(stock_list, weight):
    cov = pd.read_csv(utils.DATA_DIR + "/COV30", index_col=0)
    cov = cov.loc[stock_list, stock_list]
    weight = np.array(weight)
    v = weight.T.dot(cov).dot(weight)
    return v


if __name__ == "__main__":
    path1 = "/Users/mac/PycharmProjects/RLPortfolio(Safe DQN portaction for COLAB)/Data/AAPL"
    path2 = "/Users/mac/PycharmProjects/RLPortfolio(Safe DQN portaction for COLAB)/Data/BIDU"
    path3 = "/Users/mac/PycharmProjects/RLPortfolio(Safe DQN portaction for COLAB)/Data/COST"

    train_data, test_data = get_data(path1,
                 train_date_start="20090101",
                 train_date_end="20180101",
                 test_date_start="20180101",
                 test_date_end=None)

    print(train_data.head())
    # print(test_data.shape)