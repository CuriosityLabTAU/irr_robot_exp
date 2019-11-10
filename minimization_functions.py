from general_quantum_operators import *
from copy import deepcopy
from scipy.optimize import minimize
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from itertools import product
# from scipy.optimize import dual_annealing

def fun_to_minimize_all(h_all, real_P, psi_0, return_all=False):
    '''
    :param h_all: [h_a, h_b, h_c, h_d, gamma_]
    :param real_P: dictionary of the real probabilities for the first 2 question
    :return:
    '''
    p_a, p_b, p_ab, e_a, e_b, e_ab = get3probs(h_all, real_P, psi_0, qs=['A', 'B', 'A_B'])

    # propagate psi
    H_ = compose_H([h_all[0], h_all[1], h_all[-1]], all_q = [0,1])
    psi_1 = get_psi(H_, psi_0)

    p_c, p_d, p_cd, e_c, e_d, e_cd = get3probs([h_all[2],h_all[3],h_all[-1]], real_P, psi_1, qs=['C', 'D', 'C_D'])

    err_ = rmse([p_a, p_b, p_ab, p_c, p_d, p_cd], list(real_P.values()))
    if return_all:
        return p_a, p_b, p_ab, p_c, p_d, p_cd, e_a, e_b, e_ab, e_c, e_d, e_cd, H_, psi_1
    else:
        return err_

def get3probs(h_all, real_P, psi_0, qs = ['A', 'B', 'A_B']):
    ''' helping function for fun_to_minimize_all
    :return: 3 probs + errors
    '''
    e_a, p_a = fun_to_minimize_const_gamma(h_all[0], real_P[qs[0]], psi_0,['x', None, None], [0,1], '0', 4)
    e_b, p_b = fun_to_minimize_const_gamma(h_all[1], real_P[qs[1]], psi_0, [None, 'x', None], [0,1], '1', 4)
    e_ab, p_ab = fun_to_minimize_const_gamma(h_all[-1], real_P[qs[-1]], psi_0, [None, None, 'x'], [0,1], 'C', 4)
    return p_a, p_b, p_ab, e_a, e_b, e_ab

def fun_to_minimize(h_, real_p_, psi_0, all_h, all_q, all_P, n_qubits=2, h_mix_type = 0):
    # all_h = ['x', h_b, h_ab], [h_a, None, h_ab], [h_a, h_b, None]
    # all_q = [q1, q2] = [0,3] --> AD
    # all_P = '0' --> P_q1, '1' --> P_q2, 'C' --> P_q1 * P_q2, 'D' --> P_q1 + P_q2 - P_q1 * P_q2

    full_h = [h_[0] if type(v) is type('x') else v for v in all_h] # replace the 'x' with the minimization parameter
    p_ = get_general_p(full_h, all_q, all_P, psi_0, n_qubits, h_mix_type = h_mix_type)
    err_ = rmse(p_, real_p_)
    return err_

def fun_to_minimize_const_gamma(h_, real_p_, psi_0, all_h, all_q, all_P, n_qubits=2, h_mix_type = 0):
    # all_h = ['x', h_b, h_ab], [h_a, None, h_ab], [h_a, h_b, None]
    # all_q = [q1, q2] = [0,3] --> AD
    # all_P = '0' --> P_q1, '1' --> P_q2, 'C' --> P_q1 * P_q2, 'D' --> P_q1 + P_q2 - P_q1 * P_q2

    full_h = [h_ if type(v) is type('x') else v for v in all_h] # replace the None with the minimization parameter
    p_ = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4, h_mix_type = h_mix_type)
    err_ = rmse(p_, real_p_)
    return err_, p_

def fun_to_minimize_grandH(x_, all_q, all_data, h_mix_type = 0, fal ='C'):
    grand_U = U_from_H(grandH_from_x(x_, fal))

    err_ = []
    for data in all_data.values():
        psi_0 = np.dot(grand_U, data[1]['psi'])

        h_a = data[2]['h_q'][str(all_q[0])]
        p_a_calc = get_general_p(full_h=[h_a, None, None],
                                 all_q=all_q,
                                 all_P='0', psi_0=psi_0, n_qubits=4,
                                 h_mix_type = h_mix_type)
        p_a = data[2]['p_a']
        err_.append((p_a_calc - p_a) ** 2)

        h_b = data[2]['h_q'][str(all_q[1])]
        p_b_calc = get_general_p(full_h=[None, h_b, None],
                                 all_q=all_q,
                                 all_P='1', psi_0=psi_0, n_qubits=4,
                                 h_mix_type=h_mix_type)
        p_b = data[2]['p_b']
        err_.append((p_b_calc - p_b) ** 2)

    return np.sqrt(np.mean(err_))


def general_minimize(f, args_, x_0, method = 'L-BFGS-B', bounds = None):
    min_err = 100.0
    best_result = None
    num_of_minimizations = 1
    x_0r = []

    ### set the bounds of the minimization problem
    if (bounds is None) and (method == 'L-BFGS-B'):
        bounds = np.ones([x_0.shape[0], 2])
        bounds[:, 0] = -1

    for i in range(num_of_minimizations):
        # x_0r.append(np.random.randint(2, size=x_0.shape) * 2.0 - 1.0)
        # x_0r.append(np.random.random(size = x_0.shape) * 2.0 - 1.0)
        # x_0_rand = x_0r[np.random.randint(2)]
        # res_temp = minimize(f, x_0_rand, args=args_, method=method, bounds=bounds, options={'disp': False})
        res_temp = minimize(f, x_0, args=args_, method=method, bounds=bounds, options={'disp': False})

        if res_temp.fun < min_err:
            min_err = res_temp.fun
            best_result = deepcopy(res_temp)

    return best_result

# def general_minimize(f, args_, x_0):
#     '''minimization using threading'''
#     print('threading')
#     min_err = 100.0
#     best_result = None
#     num_of_minimizations = 100
#
#     # Creating the product for the minimization using pool and starmap.
#     prod = product([f], [np.random.random(x_0.shape[0]) * 2.0 - 1.0 for i in range(num_of_minimizations)], [args_], ['SLSQP'],
#                    [None], [None], [None], [None], [()], [1e-6], [None], [{'disp': False, 'maxiter': 100}])
#
#     # Threading the minimizations on the CPUs
#     # with ThreadPool(cpu_count()) as p:
#     p = ThreadPool(cpu_count())
#     results = p.starmap(minimize, prod)
#     p.close()
#     p.join()
#
#     # Find the best result.
#     for res_temp in results:
#         if res_temp.fun < min_err:
#             min_err = res_temp.fun
#             best_result = deepcopy(res_temp)
#
#     return best_result