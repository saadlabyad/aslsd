# License: BSD 3 clause
import numpy as np
from scipy.special import jv
from tqdm import tqdm

from aslsd.utilities import useful_functions as uf
from aslsd.utilities import useful_numerics as un


def kepler(theta, epsilon=.5):
    return theta-epsilon*np.sin(theta)


def extend_inv_kepler_func(inv_kepler_func, **kwargs):
    def general_inverse_kepler(y, epsilon=.5):
        return_float = False
        if not uf.is_array(y):
            y = np.array([y])
            return_float = True
        len_y = len(y)
        theta = np.zeros(len_y)

        mask_0pi = np.where((y >= 0.) & (y < np.pi))[0]
        mask_pi2pi = np.where((y >= np.pi) & (y < 2.*np.pi))[0]
        mask_2pi3pi = np.where((y >= 2.*np.pi) & (y < 3.*np.pi))[0]

        if len(mask_0pi) > 0:
            y_vals = y[mask_0pi]+0.
            theta[mask_0pi] = inv_kepler_func(y_vals, epsilon=epsilon,
                                              **kwargs)
        if len(mask_pi2pi) > 0:
            y_vals = 2.*np.pi-y[mask_pi2pi]
            # Check
            Q = np.where((y_vals < 0.) | (y_vals > np.pi))[0]
            if len(Q) != 0:
                print('Wrong scale')
                print('len(Q)', len(Q))
                print('Q', Q)
                print('y_vals[Q[0]]', y_vals[Q[0]])
            # End check
            theta[mask_pi2pi] = 2.*np.pi-inv_kepler_func(y_vals,
                                                         epsilon=epsilon,
                                                         **kwargs)
        if len(mask_2pi3pi) > 0:
            y_vals = y[mask_2pi3pi]-2.*np.pi
            # Check
            Q = np.where((y_vals < 0.) | (y_vals > np.pi))[0]
            if len(Q) != 0:
                print('Wrong scale')
                print(len(Q))
            # End check
            theta[mask_2pi3pi] = 2.*np.pi+inv_kepler_func(y_vals,
                                                          epsilon=epsilon,
                                                          **kwargs)

        if return_float:
            return theta[0]
        else:
            return theta
    return general_inverse_kepler


# =============================================================================
# Reference values
# =============================================================================

def get_kepler_refs(epsilon=.5):
    ref_theta = np.array([0., np.pi/6., np.pi/4., np.pi/3., np.pi/2.,
                          2.*np.pi/3., np.pi])
    ref_y = np.array([0., np.pi/6.-.5*epsilon, np.pi/4.-epsilon/np.sqrt(2.),
                      np.pi/3.-epsilon*(np.sqrt(3)/2.), 0.5*np.pi-epsilon,
                      (2./3.)*np.pi-epsilon*(np.sqrt(3)/2.), np.pi])
    return ref_theta, ref_y


def refine_kepler_interval(theta_l, theta_r, epsilon=.5, n_res=10**2):
    vec_theta = np.linspace(theta_l, theta_r, n_res)
    vec_y = kepler(vec_theta, epsilon=epsilon)
    return vec_theta, vec_y


def refine_kepler_refs(epsilon=.5, ref_theta=None, ref_y=None, n_res=10**2):
    if ref_theta is None:
        ref_theta, ref_y = get_kepler_refs(epsilon=epsilon)
    L_theta = []
    L_y = []
    for ix in range(len(ref_theta)-1):
        theta_l = ref_theta[ix]
        theta_r = ref_theta[ix+1]
        vec_theta, vec_y = refine_kepler_interval(theta_l, theta_r,
                                                  epsilon=epsilon, n_res=n_res)
        L_theta.extend(list(vec_theta[:-1]))
        L_y.extend(list(vec_y[:-1]))
    L_theta.append(ref_theta[-1])
    L_y.append(ref_y[-1])
    return np.array(L_theta), np.array(L_y)


def kepler_initialiser(y, epsilon=.5, ref_theta=None, ref_y=None, n_res=10**2,
                       start_type='middle', rng=None, seed=1234):
    if ref_theta is None:
        ref_theta, ref_y = get_kepler_refs(epsilon=epsilon)
    f_args = {'epsilon': epsilon}
    L_theta, L_y = un.refine_inversion_refs(ref_theta, ref_y, kepler,
                                            f_args=f_args, n_res=n_res)
    theta_0 = un.initialise_grid(y, L_theta, L_y, start_type=start_type,
                                 rng=rng, seed=seed)
    return theta_0


# =============================================================================
# Newton's method
# =============================================================================
def get_newton_ratio(y, theta, epsilon=.5):
    return (theta-epsilon*np.sin(theta)-y)/(1.-epsilon*np.cos(theta))


def get_inv_kepler_newton(y, epsilon=.5, error_func=None, theta_0=None,
                          initialiser=None, init_args=None, n_res=10**3,
                          start_type='middle',
                          rng=None, seed=1234, n_iter=10**5, tol=10**-15,
                          verbose=False):
    kwargs = {'epsilon': epsilon}
    init_args = {'epsilon': epsilon, 'n_res': n_res, 'start_type': start_type,
                 'rng': rng, 'seed': seed}
    if initialiser is None:
        initialiser = kepler_initialiser
    theta = un.get_newton(y, ratio_func=get_newton_ratio,
                          error_func=error_func, theta_0=theta_0,
                          initialiser=initialiser, init_args=init_args,
                          rng=rng, seed=seed,
                          lower_bound=0.,
                          upper_bound=np.pi, n_iter=n_iter,
                          tol=tol, verbose=verbose, **kwargs)
    return theta


# =============================================================================
# Analytical
# =============================================================================
def make_kapetyn_prefactors(n_iter, epsilon=.5, verbose=False):
    prefactors = np.zeros(n_iter+2)
    for n in tqdm(range(1, n_iter+1), disable=not verbose):
        prefactors[n] = (2./n)*jv(n, n*epsilon)
    return prefactors


def get_inv_kepler_kapetyn(y, epsilon=.5, prefactors=None,
                           error_func=None,
                           n_iter=10**4,
                           tol=10**-5, verbose=False):
    if prefactors is None:
        prefactors = make_kapetyn_prefactors(n_iter, epsilon=epsilon,
                                             verbose=False)
    if error_func is None:
        error_func = un.abs_error_func
    return_float = False
    if not uf.is_array(y):
        y = np.array([y])
        return_float = True
    len_y = len(y)

    theta = y+0.
    active_ixs = np.arange(len_y)
    for n in tqdm(range(1, n_iter+1), disable=not verbose):
        # theta_prev = theta+0.
        theta[active_ixs] += prefactors[n]*np.sin(n*y[active_ixs])
        # active_ixs = np.where((error_func(theta, theta_prev) >= tol))[0]
        # if uf.is_empty(active_ixs):
        #     if verbose:
        #         print('Method converged after ', n, ' iterations.')
        #     break

    if return_float:
        return theta[0]
    else:
        return theta
