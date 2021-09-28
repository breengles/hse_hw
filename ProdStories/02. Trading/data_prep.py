from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesResampler


def get_dataset(data, data_prep, feature):
    # non-eq.
    ts = []
    for _, values in data.groupby("session_id")[feature]:
        if values.shape[0] > 1:
            ts.append(values.tolist())

    train = to_time_series_dataset(ts)
    train = TimeSeriesResampler(sz=train.shape[1]).fit_transform(train)

    # eq.
    ts = []
    for _, values in data_prep.groupby("session_id")[feature]:
        ts.append(values.tolist())

    train_prep = to_time_series_dataset(ts)

    return train, train_prep


def get_platform_specific_dataset(data, data_prep, feature):
    platform_1 = data[data.platform_id == 1]
    platform_2 = data[data.platform_id == 2]

    platform_1_prep = data_prep[data_prep.platform_id == 1]
    platform_2_prep = data_prep[data_prep.platform_id == 2]

    ts = []
    for _, values in platform_1.groupby("session_id")[feature]:
        ts.append(values.tolist())

    train_1 = to_time_series_dataset(ts)
    train_1 = TimeSeriesResampler(sz=train_1.shape[1]).fit_transform(train_1)

    ts = []
    for _, values in platform_2.groupby("session_id")[feature]:
        ts.append(values.tolist())

    train_2 = to_time_series_dataset(ts)
    train_2 = TimeSeriesResampler(sz=train_2.shape[1]).fit_transform(train_2)

    ts = []
    for _, values in platform_1_prep.groupby("session_id")[feature]:
        ts.append(values.tolist())

    train_1_prep = to_time_series_dataset(ts)

    ts = []
    for _, values in platform_2_prep.groupby("session_id")[feature]:
        ts.append(values.tolist())

    train_2_prep = to_time_series_dataset(ts)

    return train_1, train_2, train_1_prep, train_2_prep
