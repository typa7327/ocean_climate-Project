import matplotlib.pyplot as plt

def plot_timeseries(ts, title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ts.plot(ax=ax)
    if title:
        ax.set_title(title)
    if xlabel:
         ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=150)
    print(f"Saved {title}.png")
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