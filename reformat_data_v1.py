# from hamiltonian_prediction import *
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from hamiltonian_prediction import *

import timeit

### questions organizer dictionary
questions = {'conj': {'Q2': [0, 1],
                      'Q4': [2, 3],
                      'Q6': [0, 3]}, # art
                      # 'Q6': [1, 3]}, #suspects
             'trap': {'Q8': 3},} # suspects = 2, art = 3
questions_fal = {'Q2': 1,
                 'Q4': 1,
                 'Q6': 1,}


### which options correspond to which qubits
# questions_options = {
#    'Q2' : {'pa':{'0': 1},
#            'pb':{'1': 2},
#            'pab':{'01': 3}},
#    'Q4': {'pa':{'2': 1},
#           'pb':{'3': 2},
#           'pab':{'23': 3}},
#    'Q6' : {'pa':{'1': 1},
#            'pb':{'3': 2},
#            'pab':{'13': 3}},
#}

### for art
questions_options = {
    'Q2' : {'pa':{'0': 1}, # realisitic
            'pb':{'1': 4}, # famouse
            'pab':{'01': 2}},
    'Q4': {'pa':{'2': 1}, # express
           'pb':{'3': 2}, # young
           'pab':{'23': 3}},
    'Q6': {'pa': {'0': 2}, # real
           'pb': {'3': 1}, # young
           'pab': {'03': 3}},
}

### init {probs} state in each question
prob_q = {'0' : 0, '1' : 0, '2' : 0, '3' : 0}

### init {h} state in each question
hq_q = {'0' : 0, '1' : 0, '2' : 0, '3' : 0,
        '01' : 0, '02' : 0, '03' : 0,
        '12' : 0, '13' : 0, '23' : 0,
        }

qubits_dict = {0:'a', 1:'b', 2:'c', 3:'d'}
fal_dict = {1:'C', 2: 'D'}

def q_qubits_fal(q):
    if q in list(questions['conj'].keys()):
        all_q = questions['conj'][q]
        fal = 'C'
    elif q in list(questions['disj'].keys()):
        all_q = questions['disj'][q]
        fal = 'D'
    return all_q, fal


def reformat_data_from_qualtrics(path):
    '''reformat the data from qualtrics to cleaner dataframe'''
    ### load the file
    raw_df = pd.read_csv(path)
    clms = raw_df.columns
    raw_df = raw_df.iloc[2:]

    ### clear users that fail the trap question
    ## change the range from qualtrics to [0,6]
    # vd = dict(zip(np.sort(raw_df[list(questions['trap'].keys())[0]].astype('float').unique()), np.arange(6)))
    # raw_df[list(questions['trap'].keys())[0]] = raw_df[list(questions['trap'].keys())[0]].astype('float').replace(vd)
    raw_df = raw_df[raw_df[list(questions['trap'].keys())[0]].astype('float') == list(questions['trap'].values())[0]]

    ### order of the questions
    ### choose the columns of the order
    rand_qs = ['Q6']
    rand_qs = [x + '_order' for x in rand_qs]
    order_cls = raw_df[clms[clms.str.contains('FL_')]]
    renaming_dict = dict(zip(order_cls, rand_qs))
    raw_df.rename(columns=(renaming_dict), inplace=True)

    cnames = []

    ### questions with fallacies
    all_cls = list(questions['conj'].keys())

    ### subsample the columns that i need
    for q in all_cls:
        cnames = cnames + list(clms[clms.str.contains(q)])

    raw_df = raw_df[cnames+ ['survey_code']]
    clms = raw_df.columns

    ### match option with which qubit and probability it is
    q_dict = {}
    # probs = ['pa','pb','pab']
    for q in all_cls:
        for i, (p, qd) in enumerate(questions_options[q].items()):
            qubit = list(qd.keys())[0]
            option = list(qd.values())[0]
            current_name = q + '_' + str(option)
            new_name = q + '_' + 'q' + str(qubit)+ '_' + p + '_'
            q_dict[current_name] = new_name

    raw_df.rename(columns=(q_dict), inplace=True)

    # raw_df = raw_df[list(q_dict.values()) + id_qs + list(raw_df.columns[raw_df.columns.str.contains('order')])]

    raw_df[list(q_dict.values())] = raw_df[list(q_dict.values())].astype('float') / 100
    # raw_df[list(q_dict.values())] = np.random.random(raw_df[list(q_dict.values())].shape)

    ### which question was third
    raw_df['q3'] = 'Q6'

    raw_df.to_csv('data/processed_data/clear_df.csv', index = False)

    return raw_df
def get_user_question_probs(df, question, probs = ['A', 'B']):
    '''
    ### get the real probabilities of user i for question 'Q2'
    :param df: data frame per user
    :param question: which question
    :param probs: which probabilities in the question
    :return:
    '''
    d = df[df.columns[df.columns.str.contains(question)]].reset_index(drop=True)
    p_real = {
        probs[0]: d[d.columns[d.columns.str.contains('pa_')]].values.flatten()[0],
        probs[1]: d[d.columns[d.columns.str.contains('pb_')]].values.flatten()[0],
        probs[0]+'_'+probs[1]: d[d.columns[d.columns.str.contains('pab_')]].values.flatten()[0]
    }
    return p_real


def calc_first2questions(df):
    ### calculate all the parameters and psi for the first 2 questions
    all_data = {}
    for ui, u_id in enumerate(df['survey_code'].unique()):
        start = timeit.default_timer()

        ### init psi
        psi_0 = uniform_psi(n_qubits=4)

        ### the data of specific user
        d0 = df[(df['survey_code'] == u_id)]

        ### combine the 6 probabilities for the first 2 questions to a single dictionary
        p_real = get_user_question_probs(d0, question = 'Q2', probs=['A', 'B']) # real probs of 1st question
        p_real_2 = get_user_question_probs(d0, question = 'Q4', probs=['C', 'D']) # real probs of 2nd question
        p_real.update(p_real_2)

        ### get all the data for the first 2 questions together
        # sub_data[p_id] = get_question_H_constant_gamma(psi_0)
        sub_data = get_question_H_constant_gamma(psi_0, p_real)

        p_real_3 = get_user_question_probs(d0, question = 'Q4', probs=['A', 'D']) # real probs of 3rd question
        ### for ART
        h_all = [sub_data[1]['h_q']['0'], sub_data[1]['h_q']['3'], sub_data[1]['h_q']['01']]
        p_a, p_d, p_ad, e_a, e_d, e_ad = get3probs(h_all, p_real_3, psi_0, qs=['A', 'D', 'A_D'])
        sub_data['pred_errors'] = [e_a, e_d, e_ad]
        sub_data['pred_probs3'] = [p_a, p_d, p_ad]

        ### errors from pre
        # sub_data['pre_errors'] = [rmse(p_a, p_real['A']), rmse(p_d, p_real['D'])]

        ### append current user to the dict that contains all the data
        all_data[u_id] = sub_data

        stop = timeit.default_timer()
        print('user %d/%d: ' %(ui + 1, len(df['survey_code'].unique())), stop - start)

    ### calc all errors:
    p_am = []
    p_dm = []
    p_adm = []
    for u_id, tu in all_data.items():
        p_am.append(tu[0]['p_a'][0])
        p_dm.append(tu[1]['p_b'][0])
    p_am = np.array(p_am).mean()
    p_dm = np.array(p_dm).mean()

    ### save dict with np
    np.save('data/processed_data/all_data_dict_gamma.npy', all_data)

    print(''' 
    ================================================================================
    || Done calculating {h_i} for all questions for all users.                    || 
    || Data was saved to: data/processed_data/all_data_dict.npy                   || 
    ================================================================================''')

    return all_data

def calc_all_questions_constant_gamma(df):
    ### calculate all the parameters and psi for the first 2 questions

    all_data = {}
    for ui, u_id in enumerate(df['survey_code'].unique()):
        start = timeit.default_timer()

        ### init psi
        psi_0 = uniform_psi(n_qubits=4)
        sub_data = {
            'h_q': {}
        }

        ### the data of specific user
        d0 = df[(df['survey_code'] == u_id)]

        ### run on all questions
        for p_id, q in enumerate(['Q2', 'Q4', 'Q6']):
            d = d0.copy()

            ### take the real probs of the user
            d = d[d.columns[d.columns.str.contains(q)]].reset_index(drop=True)
            p_real = {
                'A': d[d.columns[d.columns.str.contains('pa_')]].values,
                'B': d[d.columns[d.columns.str.contains('pb_')]].values,
                'A_B': d[d.columns[d.columns.str.contains('pab_')]].values
            }

            ### is the third question is conj/ disj
            all_q, fal = q_qubits_fal(q)

            sub_data[p_id] = get_question_H(psi_0, all_q, p_real, fallacy_type=fal)

            psi_0 = sub_data[p_id]['psi']

            if p_id == 0:
                sub_data[p_id]['h_q'] = hq_q.copy()
                sub_data[p_id]['prob_q'] = prob_q.copy()
            else:
                sub_data[p_id]['h_q'] = sub_data[p_id-1]['h_q'].copy()
                sub_data[p_id]['prob_q'] = sub_data[p_id-1]['prob_q'].copy()

            ### update the {h} from the most recent question.
            sub_data[p_id]['h_q'][str(all_q[0])] = sub_data[p_id]['h_a']
            sub_data[p_id]['h_q'][str(all_q[1])] = sub_data[p_id]['h_b']
            sub_data[p_id]['h_q'][str(all_q[0])+str(all_q[1])] = sub_data[p_id]['h_ab']

            ### update the {probs} from the most recent question.
            sub_data[p_id]['prob_q'][str(all_q[0])] = p_real['A']
            sub_data[p_id]['prob_q'][str(all_q[1])] = p_real['B']

        ### append current user to the dict that contains all the data
        all_data[u_id] = sub_data

        stop = timeit.default_timer()
        print('user %d/%d: ' %(ui + 1, len(df['survey_code'].unique())), stop - start)

    ### save dict with np
    np.save('data/processed_data/all_data_dict.npy', all_data)

    print(''' 
    ================================================================================
    || Done calculating {h_i} for all questions for all users.                    || 
    || Data was saved to: data/processed_data/all_data_dict.npy                   || 
    ================================================================================''')

    return all_data

def calculate_all_data_I():
    '''calculate prediction errors an all questions'''

    ### load the dataframe containing all the data
    raw_df = pd.read_csv('data/new_code/clear_df.csv')
    raw_df.rename({'survey_code': 'userID'}, axis=1, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)

    ### loading all the data of all the questions
    all_data = np.load('data/new_code/all_data_dict.npy').item()

    ### creating a dataframe to save all the predictions error --> for specific question group by 'qn' --> agg('mean')
    prediction_errors = pd.DataFrame()

    ### run on all users all questions but only I !!!
    for u_id, tu in all_data.items():
        start = timeit.default_timer()

        ### running from the 2nd question
        for pos, qn in enumerate(tu['qs'][1:]):
            p_id = pos + 1
            all_q, fal = q_qubits_fal(qn)

            temp = {}
            temp['id'] = [u_id]  # suer id
            temp['qn'] = [qn]    # which question
            temp['pos'] = [p_id] # question position

            temp['q1'] = [all_q[0]] # 1st qubit
            temp['q2'] = [all_q[1]] # 2nd qubit

            q1 = 'p_' + qubits_dict[temp['q1'][0]] # 1st probability
            q2 = 'p_' + qubits_dict[temp['q2'][0]] # 2nd probability
            q12 = 'p_' + qubits_dict[temp['q1'][0]] + qubits_dict[temp['q2'][0]]

            ### psi after the previous question
            psi_0 = tu[p_id-1]['psi']

            ### all probabilities from previous questions
            temp['p_a'] = [tu[p_id-1]['prob_q']['0']]
            temp['p_b'] = [tu[p_id-1]['prob_q']['1']]
            temp['p_c'] = [tu[p_id-1]['prob_q']['2']]
            temp['p_d'] = [tu[p_id-1]['prob_q']['3']]

            ### the probs that appear in the current question, taken from previous questions
            temp['p_a_pre'] = temp[q1]
            temp['p_b_pre'] = temp[q2]

            ### real probabilities in the current question
            temp[q1 + '_current'] = [tu[p_id]['p_a'][0]]
            temp[q2 + '_current'] = [tu[p_id]['p_b'][0]]

            ### take the most updated {h} from previous question that appear in current question
            h_a = [tu[p_id-1]['h_q'][str(int(temp['q1'][0]))], None, None]
            h_b = [None, tu[p_id-1]['h_q'][str(int(temp['q2'][0]))], None]

            ### predicted probabilities with I
            temp['p_a_pred_I'] = [get_general_p(h_a, all_q, '0', psi_0, n_qubits=4).flatten()[0]]
            temp['p_b_pred_I'] = [get_general_p(h_b, all_q, '1', psi_0, n_qubits=4).flatten()[0]]

            ### calculate the error from the previous probabilities with NO U.
            temp['p_a_err_real_pre'] = [(temp[q1 + '_current'][0] - temp['p_a_pre'][0]) ** 2]
            temp['p_b_err_real_pre'] = [(temp[q2 + '_current'][0] - temp['p_b_pre'][0]) ** 2]

            temp['p_a_err_real_pre_abs'] = [np.abs(temp[q1 + '_current'][0] - temp['p_a_pre'][0])]
            temp['p_b_err_real_pre_abs'] = [np.abs(temp[q2 + '_current'][0] - temp['p_b_pre'][0])]

            ### calculate the error from the full 4 qubits state with I
            temp['p_a_err_real_I'] = [(temp[q1 + '_current'][0] - temp['p_a_pred_I'][0]) ** 2]
            temp['p_b_err_real_I'] = [(temp[q2 + '_current'][0] - temp['p_b_pred_I'][0]) ** 2]

            temp['p_a_err_real_I_abs'] = [np.abs(temp[q1 + '_current'][0] - temp['p_a_pred_I'][0])]
            temp['p_b_err_real_I_abs'] = [np.abs(temp[q2 + '_current'][0] - temp['p_b_pred_I'][0])]

            ### calculate the error from uniform
            temp['p_a_err_real_uniform'] = [(temp[q1 + '_current'][0] - .5) ** 2]
            temp['p_b_err_real_uniform'] = [(temp[q2 + '_current'][0] - .5) ** 2]

            temp['p_a_err_real_uniform_abs'] = [np.abs(temp[q1 + '_current'][0] - .5)]
            temp['p_b_err_real_uniform_abs'] = [np.abs(temp[q2 + '_current'][0] - .5)]

            prediction_errors = pd.concat([prediction_errors, pd.DataFrame(temp)], axis=0)

        stop = timeit.default_timer()

        print('user running time: ', stop - start)

    np.save('data/new_code/all_data_dict.npy', all_data)

    prediction_errors.set_index('id', inplace=True)
    prediction_errors.to_csv('data/new_code/prediction_errors.csv')  # index=False)


def plot_errors(df):
    '''Boxplot of the errors per question type.
    Also calculate statistical difference between groups.'''

    ### list of the columns of the errors
    for col in list(df.columns):
        try:
            df[col] = df[col].str.replace('[', '').str.replace(']', '').astype('float')
        except:
            pass

    print('======> errors: printing only for questions in position 2')
    df = df[df['pos'] == 2]
    print(np.sqrt(df[df.columns[df.columns.str.contains('err')]].mean()).sort_values())


    ### errors descriptive data frame
    a = df[df.columns[df.columns.str.contains('p_a')]]
    a = a[a.columns[~a.columns.str.contains('p_ab')]]

    b = df[df.columns[df.columns.str.contains('p_b')]]

    b.columns = a.columns

    a = a.append(b)
    a = a[a.columns[a.columns.str.contains('err')]]

    a.reset_index(inplace=True, drop=True)

    aa = np.sqrt(a.mean())
    print(aa.sort_values())

    print()


def main():
    # reformat, calc_questions = True , False
    reformat, calc_questions = False, True
    # reformat, calc_questions = False, False

    # calc_errs = True
    calc_errs = False

    if reformat:
        # raw_df = reformat_data_from_qualtrics('data/raw_data/calc_u_suspects.csv')
        raw_df = reformat_data_from_qualtrics('data/raw_data/calc_u_art.csv')
    else:
        raw_df = pd.read_csv('data/processed_data/clear_df.csv')

    if calc_questions:
        calc_first2questions(raw_df) ### calculate all the data

    ### calcualte and predict erros of I
    ### TODO: delete before submting the paper!
    # if calc_errs:
    #     calculate_all_data_I() ### calculate error predictions
    # else:
    #     prediction_errors = pd.read_csv('data/new_code/prediction_errors.csv')
    #     plot_errors(prediction_errors)


if __name__ == '__main__':
    main()
