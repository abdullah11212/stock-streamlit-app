import numpy as np
from scipy.optimize import minimize

def heston_model(data):
    data['Return'] = 100 * data['Price'].pct_change()
    data.dropna(inplace=True)

    def loglike(params, returns):
        kappa, theta, sigma_v, rho, v0 = params
        dt = 1/252
        v = v0
        ll = 0
        for r in returns:
            dv = kappa*(theta - v)*dt + sigma_v*np.sqrt(max(v,1e-8))*np.random.normal()
            v = max(v + dv, 1e-8)
            ll += -0.5*(np.log(2*np.pi*v*dt) + (r**2)/(v*dt))
        return -ll

    res = minimize(loglike, [1,0.02,0.2,-0.3,0.02], args=(data['Return'],), method="Nelder-Mead")
    kappa, theta, sigma_v, rho, v0 = res.x

    v_path, v = [], v0
    for _ in range(len(data)):
        dv = kappa*(theta - v)/252 + sigma_v*np.sqrt(v)*np.random.normal()
        v = max(v + dv, 1e-8)
        v_path.append(np.sqrt(v))

    data['Volatility'] = v_path

    return {
        "kappa": kappa,
        "theta": theta,
        "sigma_v": sigma_v,
        "rho": rho,
        "v0": v0
    }, v_path, data
