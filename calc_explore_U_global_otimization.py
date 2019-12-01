import random
import numpy as np
import pandas as pd

from hamiltonian_prediction import *
from reformat_data import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import seaborn as sns
from scipy import stats

from itertools import combinations

from RegscorePy import bic # https://pypi.org/project/RegscorePy/
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
import nevergrad as ng
from termcolor import cprint

global train_q_data_qn

# psi_dyn = np.dot(U, psi_0)

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

qubits_dict = {0:'a', 1:'b', 2:'c', 3:'d'}
fal_dict = {1:'C', 2: 'D'}
hsc = ['h_a', 'h_b', 'h_c', 'h_d', 'h_ab', 'h_ac', 'h_ad', 'h_bc', 'h_bd', 'h_cd']
questions_fal = {'Q6': 'C'}
questions = {'conj': {'Q2': [0, 1],
                      'Q4': [2, 3],
                      # 'Q6': [1, 3],}} # suspects
                      'Q6': [0, 3],}} # art

### todo: how many paramas we have for each model?
dof_per_mode = {'mean80': 2,
                'pre'   : 1,
                'pred_U': 11,
                'pred_I': 1,
}


def ttest_or_mannwhitney(y1,y2):
    '''
    Check if y1 and y2 stand the assumptions for ttest and if not preform mannwhitney
    :param y1: 1st sample
    :param y2: 2nd sample
    :return: s, pvalue, ttest - True/False
    '''
    ttest = False

    # assumptions for t-test
    # https://pythonfordatascience.org/independent-t-test-python/#t_test-assumptions
    ns1, np1 = stats.shapiro(y1)  # test normality of the data
    ns2, np2 = stats.shapiro(y2)  # test noramlity of the data
    ls, lp = stats.levene(y1, y2)  # test that the variance behave the same
    if (lp > .05) & (np1 > .05) & (np2 > .05):
        ttest = True
        s, p = stats.ttest_ind(y1, y2)
    else:
        s, p = stats.mannwhitneyu(y1, y2)

    return s, p, ttest


def sub_sample_data(all_data, data_qn, df, users):
    '''return data'''
    for k, v in all_data.items():
        if k in users:
            data_qn[k] = deepcopy(v)
    return data_qn


def fun_to_minimize_grandH_skopt(x_):
    global train_q_data_qn
    ### suspects
    # all_q = [1,3]
    ### art
    all_q = [0,3]
    all_data = train_q_data_qn.copy()
    fal = 'C'
    grand_U = U_from_H(grandH_from_x(x_, fal))

    err_ = []
    for data in all_data.values():
        psi_0 = np.dot(grand_U, data[1]['psi'])

        h_a = data[2]['h_q'][str(all_q[0])]
        p_a_calc = get_general_p(full_h=[h_a, None, None],
                                 all_q=all_q,
                                 all_P='0', psi_0=psi_0, n_qubits=4)
        p_a = data[2]['p_a']
        err_.append((p_a_calc - p_a) ** 2)

        h_b = data[2]['h_q'][str(all_q[1])]
        p_b_calc = get_general_p(full_h=[None, h_b, None],
                                 all_q=all_q,
                                 all_P='1', psi_0=psi_0, n_qubits=4,)
        p_b = data[2]['p_b']
        err_.append((p_b_calc - p_b) ** 2)

    return np.sqrt(np.mean(err_))


def fun_to_minimize_grandH_nevergrad(x0, x1,x2,x3,x4,x5,x6,x7,x8,x9):
    global train_q_data_qn
    x_ = [x0, x1,x2,x3,x4,x5,x6,x7,x8,x9]
    ### suspects
    # all_q = [1,3]
    ### art
    all_q = [0,3]
    all_data = train_q_data_qn.copy()
    fal = 'C'
    grand_U = U_from_H(grandH_from_x(x_, fal))

    err_ = []
    for data in all_data.values():
        psi_0 = np.dot(grand_U, data[1]['psi'])

        h_a = data[2]['h_q'][str(all_q[0])]
        p_a_calc = get_general_p(full_h=[h_a, None, None],
                                 all_q=all_q,
                                 all_P='0', psi_0=psi_0, n_qubits=4)
        p_a = data[2]['p_a']
        err_.append((p_a_calc - p_a) ** 2)

        h_b = data[2]['h_q'][str(all_q[1])]
        p_b_calc = get_general_p(full_h=[None, h_b, None],
                                 all_q=all_q,
                                 all_P='1', psi_0=psi_0, n_qubits=4,)
        p_b = data[2]['p_b']
        err_.append((p_b_calc - p_b) ** 2)

    return np.sqrt(np.mean(err_))


def calculate_all_data_cross_val_kfold(with_mixing=True, min_type='global', kfold=True, gamma=False):
    '''cross validation only for the third question'''
    global hs, train_q_data_qn

    cprint('''
    starting to calculate U and it's prediction
    minimization type: %s
    kfold: %s,
    is (h_ab = _hcd) : %s''' %(min_type, str(kfold), str(gamma)),'magenta')

    if not os.path.exists('data/predictions/%s' % min_type):
        os.mkdir('data/predictions/%s'%min_type)

    ### load the dataframe containing all the data
    raw_df = pd.read_csv('data/processed_data/clear_df.csv')
    raw_df.rename({'survey_code':'userID'},axis = 1, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)

    if gamma:
        ### loading gamma data
        all_data = np.load('data/processed_data/all_data_dict_gamma.npy').item()
    else:
        ### loading all the data of the firs 3 questions
        all_data = np.load('data/processed_data/all_data_dict.npy').item()

    ### creating a dictionary with users that had the same question as the third question
    user_same_q_list = {}
    for q, g in raw_df.groupby('q3'):
        user_same_q_list[q] = g['userID']

    ### test_set
    test_users = {}

    ### instead of run number
    i = 0

    # third question
    ### creating a dataframe to save all the predictions error --> for specific question group by 'qn' --> agg('mean')
    df_prediction = pd.DataFrame()

    q_info = {}
    ### dataframe for storing the 10 {h} of U.
    df_h = pd.DataFrame(columns = ['qn', 'k_fold'] + hsc)

    ### Run on all users that have the same third question.
    for qn, user_list in user_same_q_list.items():
        user_list = np.array(user_list)

        ### define test percent size.
        # test_precent = 30./len(user_list) # how many % is 30 users from all the users.
        # user_list_train, user_list_test = train_test_split(user_list, test_size=test_precent , random_state=3)
        user_list_train=user_list.copy()


        try:
            test_users[qn] = user_list_test.copy()
        except:
            pass

        ### get which qubits and the fallacy of the question
        all_q, fal = q_qubits_fal(qn)

        ### split the users to test and train using kfold - each user will be one time in test
        if kfold:
            k = user_list_train.__len__()  # 10
        else:
            k = 2
        kf = KFold(n_splits=k)
        kf.get_n_splits(user_list_train)

        cprint('''==============================
        Analyzing question %s ''' % qn, 'magenta')  # to differentiate between qn
        # x0_i = np.zeros([10]) ### initialize x0 for the first run
        ### n randomizations of x_0
        n = 1
        np.random.seed(42)
        x_i = np.random.random(size=(n, 10)) * 2.0 - 1.0
        x_i[0, :] = 0
        for j in range(n):
            cprint('''=========== initialization of x_0 number %d============''' % j, 'blue')
            x0_i = x_i[j, :]

            for i, (train_index, test_index) in enumerate(kf.split(user_list_train)):
                cprint(
                    'currently running k_fold analysis to calculate U on question: %s, k = %d/%d.' % (qn, i + 1, k), 'blue')
                q_info[qn] = {}
                q_info[qn]['U_params_h'] = {}
                q_info[qn]['H_ols'] = {}
                train_users, test_users = user_list_train[train_index], user_list_train[test_index]
                train_q_data_qn = {}
                test_q_data_qn = {}

                train_q_data_qn = sub_sample_data(all_data, train_q_data_qn, raw_df, train_users)
                test_q_data_qn = sub_sample_data(all_data, test_q_data_qn, raw_df, test_users)

                ### check
                # train_q_data_qn = train_q_data_qn[11685720]
                # test_q_data_qn[11685720] = train_q_data_qn.copy()
                # train_q_data_qn = test_q_data_qn.copy()

                ### taking the mean of the probabilities of the 80 %
                p_a_80 = []
                p_b_80 = []
                p_ab_80 = []
                for u_id, tu in train_q_data_qn.items():
                    try:
                        p_a_80.append(tu[2]['p_a'][0])
                        p_b_80.append(tu[2]['p_b'][0])
                        p_ab_80.append(tu[2]['p_ab'][0])
                    except:
                        p_a_80.append(tu[2]['p_a'])
                        p_b_80.append(tu[2]['p_b'])
                        p_ab_80.append(tu[2]['p_ab'])
                p_a_80 = np.array(p_a_80).mean()
                p_b_80 = np.array(p_b_80).mean()
                p_ab_80 = np.array(p_ab_80).mean()

                if len(train_q_data_qn) > 0:
                    ### question qubits (-1) because if the range inside of some function
                    h_names = ['0', '1', '2', '3', '01', '23', str(all_q[0]) + str(all_q[1])]

                    # find U for each question #
                    start = time.perf_counter()  

                    ### set bounds to all parameters
                    bounds = np.ones([10, 2])
                    bounds[:, 0] = -1

                    ### local minimization
                    if min_type == 'local':
                        res_temp = general_minimize(fun_to_minimize_grandH, args_=(all_q, train_q_data_qn, 0, fal),
                                                    x_0=x0_i, method='L-BFGS-B', bounds=bounds) #, look at global optimizations at scipy.optimize
                                                    # x_0=np.zeros([10]), method='Powell', bounds=bounds) #L-BFGS-B, look at global optimizations at scipy.optimize
                                                                      # method='annealing'
                    ### global minimization
                    if min_type == 'global':
                        res_temp = gp_minimize(fun_to_minimize_grandH_skopt,  # the function to minimize
                                          dimensions=bounds,  # the bounds on each dimension of x
                                          acq_func="EI",  # the acquisition function
                                          n_calls=300,  # the number of evaluations of f
                                          n_random_starts=10,  # the number of random initialization points
                                          n_jobs=4,
                                          # noise=0.1 ** 2,  # the noise level (optional)
                                          random_state=123)  # the random seed


                    # instrum = ng.Instrumentation(ng.var.Array(10,1).bounded(-1.,1.))
                    # optimizer = ng.optimizers.OnePlusOne(instrumentation=instrum, budget=100)
                    # optimizer = ng.optimizers.PSO(instrumentation=instrum, budget=500) ### really good
                    # optimizer = ng.optimizers.PSO(instrumentation=instrum, budget=400, num_workers=20)
                    # optimizer = ng.optimizers.BO(instrumentation=instrum, budget=200)
                    # optimizer = ng.optimizers.TwoPointsDE(instrumentation=instrum, budget=200)
                    # optimizer = ng.optimizers.PSOParticle(x=instrum)
                    # recommendation = optimizer.minimize(fun_to_minimize_grandH_skopt)

                    # from concurrent import futures
                    # optimizer = ng.optimizers.OnePlusOne(instrumentation=instrum, budget=100, num_workers=5)
                    # optimizer = ng.optimizers.PSO(instrumentation=instrum, budget=100, num_workers=5)
                    # with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
                    #     recommendation = optimizer.minimize(fun_to_minimize_grandH_skopt, executor=executor, batch_mode=False)

                    # print(recommendation.args)
                    # res_temp = recommendation.args[0].flatten()

                    # print(res_temp)

                    end = time.perf_counter()  
                    cprint('question %s, U optimization took %.2f s' % (qn, end - start), 'blue')

                    ### saving all 10 {h_i} for this k fold run
                    q_info[qn]['U_params_h'][i] = [res_temp]
                    df_h = df_h.append(pd.DataFrame(
                        columns=['qn', 'k_fold'] + hsc,
                        data = [[qn, i] + list(res_temp.x)]))

                    ### calculate U from current {h_i}
                    q_info[qn]['U'] = U_from_H(grandH_from_x(res_temp.x, fal))

                    ### calculate H_AB model based on MLR/ ANN to predict p_ab
                    H_dict = {}
                    for u_id, tu in test_q_data_qn.items():
                        psi_0 = np.dot(q_info[qn]['U'], tu[1]['psi'])

                        p_real, d = sub_q_p(raw_df, u_id, 2)
                        sub_data_q = get_question_H(psi_0, all_q, p_real,
                                                    [tu[1]['h_q'][str(all_q[0])],
                                                     tu[1]['h_q'][str(all_q[1])]],
                                                    with_mixing, fallacy_type=fal)
                        tu[2]['h_q'] = tu[1]['h_q'].copy()
                        tu[2]['h_q'][str(all_q[0]) + str(all_q[1])] = sub_data_q['h_ab']
                        H_dict[u_id] = []
                        for hs in h_names:
                            H_dict[u_id].append(tu[2]['h_q'][hs])

                    df_H = pd.DataFrame.from_dict(data=H_dict, orient='index')
                    df_H.columns = ['A', 'B', 'C', 'D', 'AB', 'CD', 'target']

                    start = time.perf_counter()  
                    mtd = 'lr'  # 'ANN'
                    est = pred_h_ij(df_H, method = mtd)
                    end = time.perf_counter()  
                    cprint('question %s, h_ij using %s prediction took %.2f s' % (qn, mtd, end - start), 'blue')

                    q_info[qn]['H_ols'][i] = est

                    ### TODO: add this controls.
                    ### linear and logistic regression on for predicting the third question probs (real data).
                    ### from all the (probs) prediction --> irr: 0/1 (predicted data).
                    ### logistic regression for predicting the third question irr (real data).

                    np.save('data/predictions/%s_%s/kfold_all_data_dict_kfold_%d.npy' % (min_type, str(gamma), j), all_data)
                    np.save('data/predictions/%s_%s/kfold_UbyQ_kfold_%d.npy' % (min_type, str(gamma), j), q_info)

                    ### predict on test users
                    cprint('calculating predictions on test data', 'blue')
                    U = q_info[qn]['U']
                    for u_id, tu in test_q_data_qn.items():
                        temp = {}
                        temp['id'] = [u_id]
                        temp['qn'] = [qn]

                        temp['q1'] = [all_q[0]]
                        temp['q2'] = [all_q[1]]

                        q1 = 'p_' + qubits_dict[temp['q1'][0]]
                        q2 = 'p_' + qubits_dict[temp['q2'][0]]
                        q12 = 'p_' + qubits_dict[temp['q1'][0]] + qubits_dict[temp['q2'][0]]

                        ### psi after the 2nd question
                        psi_0 = tu[1]['psi']

                        ### propogate psi with the U of the 3rd question
                        psi_dyn = np.dot(U, psi_0)

                        ### probabilities from the 1st and 2nd question
                        try:
                            temp['p_a'] = [tu[0]['p_a'][0]]
                            temp['p_b'] = [tu[0]['p_b'][0]]
                            temp['p_c'] = [tu[1]['p_a'][0]]
                            temp['p_d'] = [tu[1]['p_b'][0]]
                        except:
                            temp['p_a'] = [tu[0]['p_a']]
                            temp['p_b'] = [tu[0]['p_b']]
                            temp['p_c'] = [tu[1]['p_c']]
                            temp['p_d'] = [tu[1]['p_d']]

                        ### probs of the current question taken from previous questions
                        temp['p_a_pre'] = temp[q1]
                        temp['p_b_pre'] = temp[q2]

                        ### real probabilities in the third question
                        try:
                            temp['p_a_real'] = [tu[2]['p_a'][0]]
                            temp['p_b_real'] = [tu[2]['p_b'][0]]
                            temp['p_ab_real'] = [tu[2]['p_ab'][0]]
                        except:
                            temp['p_a_real'] = [tu[2]['p_a']]
                            temp['p_b_real'] = [tu[2]['p_b']]
                            temp['p_ab_real'] = [tu[2]['p_ab']]

                        ### predicted probabilities with U
                        h_a = [tu[1]['h_q'][str(int(temp['q1'][0]))], None, None]
                        h_b = [None, tu[1]['h_q'][str(int(temp['q2'][0]))], None]
                        h_ad = [None, None, tu[1]['h_q']['01']]

                        temp['p_a_pred_U'] = [get_general_p(h_a, all_q, '0', psi_dyn, n_qubits=4).flatten()[0]]
                        temp['p_b_pred_U'] = [get_general_p(h_b, all_q, '1', psi_dyn, n_qubits=4).flatten()[0]]
                        temp['p_ad_pred_U'] = [get_general_p(h_ad, all_q, 'C', psi_dyn, n_qubits=4).flatten()[0]]

                        ### predicted probabilities with I
                        temp['p_a_pred_I']  = [get_general_p(h_a, all_q, '0', psi_0, n_qubits=4).flatten()[0]]
                        temp['p_b_pred_I']  = [get_general_p(h_b, all_q, '1', psi_0, n_qubits=4).flatten()[0]]
                        temp['p_ad_pred_I'] = [get_general_p(h_ad, all_q, 'C', psi_0, n_qubits=4).flatten()[0]]


                        ### predicted probabilities with mean from fold %
                        temp['p_a_mean80'] = [p_a_80]
                        temp['p_b_mean80'] = [p_b_80]
                        temp['p_ab_mean80'] = [p_ab_80]

                        # use question H to generate h_ab
                        h_names_gen = ['0', '1', '2', '3', '01', '23']
                        if with_mixing:
                            all_h = {'one': []}
                            for hs in h_names_gen:
                                all_h['one'].append(tu[2]['h_q'][hs])
                            df_H = pd.DataFrame.from_dict(data=all_h, orient='index')
                            df_H.columns = ['A', 'B', 'C', 'D', 'AB', 'CD']
                            try:
                                h_ab = q_info[qn]['H_ols'][i].predict(df_H).values[0]
                            except:
                                h_ab = q_info[qn]['H_ols'][i].predict(df_H)[0]
                        else:
                            h_ab = 0.0

                        # full_h = [tu['h_q'][str(int(temp['q1'][0]))], tu['h_q'][str(int(temp['q2'][0]))], h_ab]
                        full_h = [None, None, h_ab]
                        temp['p_ab_pred_ols_U']    = [get_general_p(full_h, all_q, fal, psi_dyn, n_qubits=4).flatten()[0]]
                        temp['p_ab_pred_ols_I'] = [get_general_p(full_h, all_q, fal, psi_0, n_qubits=4).flatten()[0]]

                        df_prediction = pd.concat([df_prediction, pd.DataFrame(temp)], axis=0)
                        cprint(df_prediction.shape, 'magenta')

    np.save('data/predictions/%s_%s/test_users.npy' % (min_type, str(gamma)), test_users)
    df_h.reset_index(inplace=True)
    df_h.to_csv('data/predictions/%s_%s/df_h.csv'% (min_type, str(gamma)))
    # df_H.to_csv('data/predictions/df_H_ols.csv')
    df_prediction.set_index('id', inplace=True)
    df_prediction.to_csv('data/predictions/%s_%s/kfold_prediction.csv' % (min_type, str(gamma)))  # index=False)
    cprint(''' 
    ================================================================================
    || Done calculating {h_i} for every qn/ kfold.                                || 
    || Predictions were saved to: data/predictions/kfold_prediction.csv           || 
    || {h_i} were saved to: data/predictions/df_h.csv                             || 
    || Dictionary of test users (per qn): saved to data/test_users.npy            || 
    || Supplementary files were saved to: kfold_all_data_dict.npy, kfold_UbyQ.npy || 
    ================================================================================''', 'magenta')

def statistical_diff_h(df_h,i = ''):
    '''
    Calculate which {h} per qn are statistically significant from zero.
    :param df_h: dataframe with the h per qn per kfold.
    :param i: when running multiple dataframes, save each with different name.
    :return: df_u90, {h} per qn that are statistically significant from zero.
    '''


    grouped_df = df_h.groupby('qn')

    sig_df = pd.DataFrame()

    for q in list(grouped_df.groups.keys()):
        temp = {}
        temp['qn'] = [q]
        for h in hsc:
            temp[h] = [0]
            print('''
            ===============
            qn = %s, h = %s
            ''' % (q, h))
            s, p = stats.ttest_1samp(grouped_df.get_group(q)[h].values, 0)
            #s, p, is_t = ttest_or_mannwhitney(grouped_df.get_group(q)[h], np.zeros(grouped_df.get_group(q)[h].shape[0]))
            # print('''
            # s = %.2f, p = %.2f, is t_test = %s''' % (s, p, str(is_t)))
            # if p < 0.05:
            # temp[h] = [grouped_df.get_group(q)[h].mean()]
            temp[h] = [grouped_df.get_group(q)[h].values[0]]

        ### which qubits are asked in the question
        if q in questions['conj'].keys():
            temp['qubits'] = str(questions['conj'][q])
        else:
            temp['qubits'] = str(questions['disj'][q])
        sig_df = sig_df.append(pd.DataFrame(temp))

    sig_df.reset_index(inplace=True, drop=True)
    sig_df.to_csv('data/predictions/sig_h_per_qn%s.csv' % i, index=0)


def predict_u90(sig_h, i='', run_train_users = True):
    '''
    Predict {p_i} per qn based only the {h} that are different from zero.
    :param sig_h: {h} that are statistically significant from zero per qn.
    :param kfold_test_users: dataframe of predictions based on I, U (not different from zero), mean80, pre and uniform.
    :param i: when running multiple dataframes, save each with different name.
    :param run_train_users: True/ False: predict on train (validation users) or on test users.
    :return: kfold_prediction: add the predictions of significant U to the dataframe.
    '''
    # a = np.load('data/predictions/test_users.npy')
    df_preds = pd.read_csv('data/predictions/kfold_prediction.csv')
    train_users = df_preds['id'].tolist()
    all_data = np.load('data/processed_data/all_data_dict.npy').item()
    all_users = list(all_data.keys())
    if run_train_users:
        test_users = all_users.copy()
    else:
        test_users = list(set(all_users) - set(train_users))

    ### load the dataframe containing all the data
    raw_df = pd.read_csv('data/processed_data/clear_df.csv')
    raw_df.rename({'survey_code':'userID'},axis = 1, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)
    raw_df.set_index('userID', inplace=True)
    raw_df['userID'] = raw_df.index

    ### find which question use the third question for each user of the test users
    test_preds = pd.DataFrame(raw_df.loc[test_users, 'q3'])
    train_preds = pd.DataFrame(raw_df.loc[train_users, 'q3'])

    test_q_data_qn = {}
    train_q_data_qn = {}

    test_q_data_qn = sub_sample_data(all_data, test_q_data_qn, raw_df, test_users)
    train_q_data_qn = sub_sample_data(all_data, train_q_data_qn, raw_df, train_users)

    sig_h.set_index('qn', inplace=True)

    df_prediction = pd.DataFrame()
    for qn in sig_h.index.unique():

        U = U_from_H(grandH_from_x(sig_h.loc[qn, hsc].values, questions_fal[qn]))

        all_q, fal = q_qubits_fal(qn)

        train_users_same_qn = list(train_preds[train_preds['q3'] == qn].index)
        test_users_same_qn = list(test_preds[test_preds['q3'] == qn].index)

        ### taking the mean of the probabilities of the 80 %
        p_a_80 = []
        p_b_80 = []
        p_ab_80 = []
        for u_id in train_users_same_qn:
            tu = all_data[u_id]
            p_a_80.append(tu[2]['p_a'][0])
            p_b_80.append(tu[2]['p_b'][0])
            p_ab_80.append(tu[2]['p_ab'][0])
        p_a_80 = np.array(p_a_80).mean()
        p_b_80 = np.array(p_b_80).mean()
        p_ab_80 = np.array(p_ab_80).mean()

        for u_id in test_users_same_qn:
            tu = all_data[u_id]

            temp = {}
            temp['id'] = [u_id]
            temp['qn'] = [qn]

            temp['q1'] = [all_q[0]]
            temp['q2'] = [all_q[1]]

            q1 = 'p_' + qubits_dict[temp['q1'][0]]
            q2 = 'p_' + qubits_dict[temp['q2'][0]]
            q12 = 'p_' + qubits_dict[temp['q1'][0]] + qubits_dict[temp['q2'][0]]

            ### psi after the 2nd question
            psi_0 = tu[1]['psi']

            ### propogate psi with the U of the 3rd question
            psi_dyn = np.dot(U, psi_0)

            ### probabilities from the 1st and 2nd question
            temp['p_a'] = [tu[0]['p_a'][0]]
            temp['p_b'] = [tu[0]['p_b'][0]]
            temp['p_c'] = [tu[1]['p_a'][0]]
            temp['p_d'] = [tu[1]['p_b'][0]]

            ### probs of the current question taken from previous questions
            temp['p_a_pre'] = temp[q1]
            temp['p_b_pre'] = temp[q2]

            ### real probabilities in the third question
            temp['p_a_real'] = [tu[2]['p_a'][0]]
            temp['p_b_real'] = [tu[2]['p_b'][0]]
            temp['p_ab_real'] = [tu[2]['p_ab'][0]]

            ### predicted probabilities with U
            h_a = [tu[1]['h_q'][str(int(temp['q1'][0]))], None, None]
            h_b = [None, tu[1]['h_q'][str(int(temp['q2'][0]))], None]

            temp['p_a_pred_U'] = [get_general_p(h_a, all_q, '0', psi_dyn, n_qubits=4).flatten()[0]]
            temp['p_b_pred_U'] = [get_general_p(h_b, all_q, '1', psi_dyn, n_qubits=4).flatten()[0]]

            ### predicted probabilities with I
            temp['p_a_pred_I'] = [get_general_p(h_a, all_q, '0', psi_0, n_qubits=4).flatten()[0]]
            temp['p_b_pred_I'] = [get_general_p(h_b, all_q, '1', psi_0, n_qubits=4).flatten()[0]]

            ### predicted probabilities with mean from fold %
            temp['p_a_mean80'] = [p_a_80]
            temp['p_b_mean80'] = [p_b_80]

            df_prediction = pd.concat([df_prediction, pd.DataFrame(temp)], axis=0)

        df_prediction.to_csv('data/predictions/10percent_predictions%s.csv' % i)  # index=False)


def compare_predictions(prediction_df, s=''):
    '''
    Compare the different models.
    + calculate the errors (y' - y)^2 between the prediction to real value --> pred_errs.csv
    :param kfold_prediction: dataframe containing all the predictions and real probabilities.
    :param i: when running multiple dataframes, save each with different name.
    :return: model_compare.csv --> dataframe of comparison between the models.
    '''

    for c in prediction_df.columns:
        try:
            prediction_df[c] = prediction_df[c].str.replace('[','').str.replace(']','').astype('float')
        except:
            pass

    ### --> from here combine the separate probabilities predictions to one.
    pa_columns = prediction_df.columns.str.contains('p_a_')
    pb_columns = prediction_df.columns.str.contains('p_b_')
    pac = prediction_df.columns[pa_columns]
    pbc = prediction_df.columns[pb_columns]
    nonrelevant_columns = prediction_df.columns[~pa_columns * ~pb_columns]

    df1 = pd.concat((prediction_df[nonrelevant_columns], prediction_df[pac]), axis = 1)
    df2 = pd.concat((prediction_df[nonrelevant_columns], prediction_df[pbc]), axis = 1)
    df2.rename(columns = dict(zip(pbc,pac)), inplace = True)

    df = pd.concat((df1, df2), axis = 0)

    df.columns = df.columns.str.replace('p_a_','')

    ### dataframe with the errors
    df_pred_errs = df.copy()

    real_prob = df['real']

    df_bic = pd.DataFrame()

    for i, pred in enumerate(list(pac.str.replace('p_a_',''))):
        if pred == 'real':
            continue
        p = dof_per_mode[pred]
        cbic = bic.bic(real_prob, df[pred], p)
        crmse = np.sqrt(mean_squared_error(real_prob, df[pred]))
        df_bic.loc[i,'bic'] = cbic
        df_bic.loc[i,'rmse'] = crmse
        df_bic.loc[i,'model'] = pred
        df_bic.loc[i,'dof'] = p
        print('model = %s | dof = %d | rmse = %.2f' % (pred, p, crmse))

        df_pred_errs['err_' + pred] = (df[pred] - real_prob) ** 2

    df_bic.to_csv('data/predictions/bic%s.csv'%s, index=0)
    df_pred_errs.to_csv('data/predictions/pred_errs%s.csv'%s, index=0)


def add_errors(df):
    '''
    :param df:
    :return:
    '''

def main():
    calcU = True
    # calcU = False

    # average_U = True
    average_U = False

    ### Conditions of the run
    if calcU:
        for gamma in [True, False]:
            for mt in ['global']:
                calculate_all_data_cross_val_kfold(min_type=mt, kfold=True, gamma=gamma)

    if average_U:
        df_h = pd.read_csv('data/predictions/df_h.csv')
        # N = 30
        N = 1
        for i in range(N):
            fn = '_%d' % i
            n = int(df_h.shape[0]/N)
            df_h_ = df_h.iloc[i*n:(i + 1)*n , :]
            statistical_diff_h(df_h_, fn)

            sig_h = pd.read_csv('data/predictions/sig_h_per_qn%s.csv' % fn)
            predict_u90(sig_h, fn, run_train_users=False)

            df_prediction = pd.read_csv('data/predictions/10percent_predictions%s.csv'% fn)  # index=False)
            compare_predictions(df_prediction, fn)

    # find the best u on train
    # check if its better than all other precitions
    # find the best u on test
    # check if its the same one as the train
    # check if its better than all other precitions
    # calc t-test between all errors, ANOVA + post_hoc
    # df_pred_errs  = pd.read_csv('data/predictions/run_on_test_users/pred_errs_%s.csv'% 6)
    print()

if __name__ == '__main__':
    main()
    # combining the dataframes
    # sig_h_tot = pd.DataFrame()
    # for i in range(30):
    #     fn = '_%d' % i
    #     sig_h = pd.read_csv('data/predictions/sig_h_per_qn%s.csv' % fn)
    #     # sig_h = pd.read_csv('data/predictions/bic%s.csv' % fn)
    #     # sig_h['run'] = i
    #     # sig_h_tot = pd.concat((sig_h_tot,sig_h), axis = 0)
    # sig_h_tot.to_csv('data/predictions/sig_h_per_qn_tot.csv',index=0)
    # # sig_h_tot.to_csv('data/predictions/bic_tot.csv', index=0)



