import numpy as np
import scipy.stats as stats

def statistical_analysis(data):
    log_returns = np.log(data['Price'] / data['Price'].shift(1)).dropna()

    mean_return = float(log_returns.mean())
    volatility = float(log_returns.std(ddof=1))
    variance = float(log_returns.var(ddof=1))
    skewness = float(log_returns.skew())

    excess_kurtosis = float(log_returns.kurt())
    pearson_kurtosis = excess_kurtosis + 3

    ks_D, ks_p = stats.kstest(log_returns, 'norm', args=(mean_return, volatility))
    shapiro_stat, shapiro_p = stats.shapiro(log_returns)
    ad_result = stats.anderson(log_returns, dist='norm')

    threshold = 3 * volatility
    data['Log_Return'] = log_returns
    data.dropna(inplace=True)
    data['Jump'] = ((data['Log_Return'] - mean_return).abs() > threshold).astype(int)

    return {
        "mean_return": mean_return,
        "volatility": volatility,
        "variance": variance,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "pearson_kurtosis": pearson_kurtosis,
        "ks_D": ks_D,
        "ks_p": ks_p,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "ad_statistic": ad_result.statistic,
        "ad_critical_values": ad_result.critical_values,
        "jump_count": int(data['Jump'].sum()),
        "jump_percent": 100 * data['Jump'].sum() / len(data),
        "data": data
    }
