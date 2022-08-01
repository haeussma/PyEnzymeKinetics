from lmfit import Parameters
from typing import Any, Dict, List, Callable, Tuple
from pyenzymekinetics.utility.initial_parameters import get_v, get_initial_Km, get_initial_vmax
from numpy import max, ndarray

model_params_dict: Dict[str, List[str]] = {
    "irrev MM": ["kcat", "Km"],
    "irrev MM with enzyme inactivation": ["kcat", "Km", "ki"]
}


class KineticModel():
    def __init__(self,
                 name: str,
                 model: Callable,
                 params: list,
                 kcat_initial: float,
                 Km_initial: float,
                 w0: Dict[str, ndarray]
                 ) -> None:

        self.name = name
        self.model = model
        self.params = params
        self.w0 = w0
        self.kcat_initial = kcat_initial
        self.Km_initial = Km_initial
        self.parameters = self.set_params(*self.params)
        self.w0 = w0
        self.result = None

    def set_params(self, *args):

        parameters = Parameters()
        parameters.add('k_cat', value=self.kcat_initial,
                       min=self.kcat_initial/100, max=self.kcat_initial*100)
        parameters.add('Km', value=self.Km_initial, min=self.Km_initial/100,
                       max=max(self.Km_initial)*1000)

        if "ki" in self.params:
            parameters.add("ki", value=0.01, min=0.0001, max=0.9999)

        if "kpi" in self.params:
            parameters.add("kpi", value=0.01, min=0.0001, max=1)

        if "a" in self.params:
            parameters.add("a", value=0.1, min=0.0001, max=0.9999)

        return parameters


def menten_irreversible(w0: tuple, t, params):
    cS, cE, cP = w0

    k_cat = params['k_cat'].value
    K_m = params['Km'].value

    dc_S = -k_cat * cE * (cS) / (K_m+cS)
    dc_E = 0
    dc_P = -dc_S

    return (dc_S, dc_E, dc_P)


def menten_irreversible_enzyme_inact(w0: tuple, t, params) -> tuple:
    cS, cE, cP = w0

    k_cat = params['k_cat'].value
    K_m = params['Km'].value
    k_i = params["ki"].value

    dc_S = -k_cat * cE * (cS) / (K_m+cS)
    dc_E = -k_i * cE
    dc_P = -dc_S

    return (dc_S, dc_E, dc_P)

# TODO: model with product inhibition
def menten_irreversible_inhibition(w0: tuple, t, params) -> tuple:
    cS, cE, cP, cI = w0

    k_cat = params['k_cat'].value
    K_m = params['Km'].value
    k_pi = params["kpi"].value

    dc_S = -k_cat * cE * (cS) / (K_m*(1+(cI / k_pi))+cS)
    dc_E = 0
    dc_P = -dc_S
    dc_I = 0

    return (dc_S, dc_E, dc_P, dc_I)


def menten_irreversible_inhibition_enz_inact(w0: tuple, t, params) -> tuple:
    cS, cE, cP, cI = w0

    k_cat = params['k_cat'].value
    K_m = params['Km'].value
    k_pi = params["kpi"].value
    k_i = params["ki"].value

    dc_S = -k_cat * cE * (cS) / (K_m*(1+(cI / k_pi))+cS)
    dc_E = -k_i * cE
    dc_P = -dc_S
    dc_I = 0

    return (dc_S, dc_E, dc_P, dc_I)

# Models if product is absorbing

def subabs_menten_irreversible(w, t, params):
    '''
    Differential equations
    Arguments:
        w: vector of state variables, here only one: w = [S]
        t: time
        params: parameters
    '''
    cS, cE, cP, cS0 = w
    
    K_m = params['Km'].value
    k_cat = params["k_cat"].value
    a = params["a"].value
    
    dc_S = (-k_cat*cE*(cS-a*cS0))/(K_m+(cS-a*cS0)/(1-a))
    dc_E = 0
    dc_P = -dc_S
    dc_S0 = 0
    

    return (dc_S, dc_E, dc_P, dc_S0)


if __name__ == "__main__":
    print("hi")
