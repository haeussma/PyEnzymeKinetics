from numpy import ndarray
import numpy as np
from dataclasses import dataclass
from scipy.integrate import odeint
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt


@dataclass
class Fit:
    """Module to perform non-linear regression for Enzyme kinetics parameter estimation"""

    substrate_conc: ndarray
    init_conc: ndarray
    time: ndarray
    enzyme_conc: float

    def __post_init__(self):
        self.params = self.set_params()
        self.result = self.fit_data()

    def set_params(self):
        kcat = self.get_initial_Km()/self.enzyme_conc
        km = self.get_initial_Km()

        params = Parameters()
        params.add('k_cat', value=kcat, min=kcat/100, max=kcat*100)
        params.add('Km', value=km, min=km/100,
                   max=np.max(self.substrate_conc)*100)

        return params

    def menten_irreversible(self, w, t, params):
        c_P, c_E, c_S0 = w

        k_cat = params['k_cat'].value
        K_m = params['Km'].value
        #kin = params["kin"].value

        dc_P = k_cat * c_E * (c_S0-c_P) / (K_m+(c_S0-c_P))
        #dc_E = -kin*c_E
        dc_E = 0
        d_cS0 = 0

        return (dc_P, dc_E, d_cS0)

    def g(self, t, w0, params):
        '''
        Solution to the ODE w'(t)=f(t,w,p) with initial condition w(0)= w0 (= [S0])
        '''
        w = odeint(self.menten_irreversible, w0, t, args=(params,))
        return w

    # Calculate residual

    def residual(self, params, t, data):

        # get dimensions of data (here we fit against 4 measurments => ndata = 4)
        ndata, nt = data.shape
        resid = 0.0 * data[:]  # initialize the residual vector

        # compute residual per data set
        for i in range(ndata):

            model = self.g(t, w0, params)  # solve the ODE with sfb.

            # get modeled product
            model = model[:, 0]

            resid[i, :] = data[i, :]-model  # compute distance to measured data

        return resid.flatten()

    def fit_data(self):
        result = minimize(self.residual, self.params, args=(
            self.time, self.substrate_conc), method='leastsq', nan_policy='omit')
        return result

    def visualize(self, save_to_path="", format="svg"):
        for i in range(self.substrate_conc.shape[0]):
            ax = plt.scatter(x=self.time, y=self.substrate_conc[i])

            # Integrate model
            s0 = self.init_conc[i]
            p0 = self.substrate_conc[i, 0]
            e0 = self.enzyme_conc

            #print(f"P: {p0}, E: {e0}, s:{s0}")

            w0 = (p0, e0, s0)

            data_fitted = self.g(self.time, w0, self.result.params)
            ax = plt.plot(self.time, data_fitted[:, 0], '-', linewidth=1)

        if len(save_to_path) != 0:
            plt.savefig(f"{save_to_path}.{format}")
            self.result.para
        return None
