import matplotlib.pyplot as plt

def plot_timeseries(da, title=None):
    da.plot()
    if title:
        plt.title(title)
    plt.show()

def plot_multiple_timeseries(ds, vars):
    for v in vars:
        ds[v].plot(label=v)
    plt.legend()
    plt.show()

def plot_lag_correlation(lag_corr, lags):
    plt.plot(lags, lag_corr)
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.show()