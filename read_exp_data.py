import ast
import os

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

participant_questions = {
    'Q0': [1],
    'Q3': [9],
    'Q6': [17],
    'Q8': [21],
}

robots_questions = {
    'Q1': [4, 5],
    'Q4': [13, 14],
    'Q7': [18, 19],
}

preference_questions = {
    'Q2': 7,
    'Q5': 16,
}

robots_colors = {
    'robot1': 'red',
    'robot2': 'blue'
}

qualtrics_quetions = {
    'Q8': 'detective_1',
    'Q9': 'art_1',
    'Q11': 'detective_2',
    'Q12': 'art_2',
    'Q15_1': 'barman',
    'Q15_2': 'analyst',
    'Q15_3': 'jury',
    'Q15_4': 'investment',
    'Q15_5': 'psychologist',
    'Q16': 'take_home'
}

path = './data/experiment_raw_data'


def sclmns(df, s):
    '''
    :param df: pandas dataframe
    :param s: string(s) the column(s) contains
    :return:
    '''
    clmns = []
    for i in s:
        clmns = clmns + list(df.columns[df.columns.str.contains(i)])
    return clmns

def calculate_corr_with_pvalues(df, method = 'pearsonr'):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')

    rho = df.corr()

    for r in df.columns:
        for c in df.columns:
            if method == 'pearsonr':
                pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
            elif method == 'spearman':
                pvalues[r][c] = round(stats.spearmanr(df[r], df[c])[1], 4)
            elif method == 'reg':
                slope, intercept, rho[r][c], pvalues[r][c], std_err = stats.linregress(x=df[r],y= df[c])

    rho = rho.round(2)
    pval = pvalues
    # create three masks
    r1 = rho.applymap(lambda x: '{}*'.format(x))
    r2 = rho.applymap(lambda x: '{}**'.format(x))
    r3 = rho.applymap(lambda x: '{}***'.format(x))
    # apply them where appropriate
    rho = rho.mask(pval <= 0.05, r1)
    rho = rho.mask(pval <= 0.01, r2)
    rho = rho.mask(pval <= 0.001, r3)

    return pvalues, rho


def dfstr2dict(cdf, df, ix, q, s):
    '''
    taking a dataframe that has a dictionary in location {index} and return cells.
    :param cdf: reformatted dataframe
    :param df: dataframe of the raw data
    :param ix: index of question
    :param q: question
    :param s: which story 1st/ 2nd
    :return:
    '''
    s = str(s)
    for i in ix:
        d = ast.literal_eval(df.loc[i, 'val'])
        if len(ix) > 1:
            r = df.loc[i, 'state']  # which robot
            r = robots_colors[r]
            try:
                d = d['probs']
                for j, k in enumerate(d):
                    cdf['s' + s + '_' + r + '_' + q + '_p_' + k[0]] = k[1] / 100

                    ### todo: calculate irrationality for robot
                    if 'and' in k[0]:
                        d1 = np.array(d)
                        pix = set([0, 1, 2]) - set([j])  # indices of single probabilities.
                        pvals = d1[:, 1].astype('float') /100. # the probabilities from the user.
                        cdf['s' + s + '_' + r + '_' + q + '_irr'] = pvals[j] - np.array(pvals)[list(pix)].min()

            except:
                d = d['rankings']
                for rnk, p in enumerate(d):
                    cdf['s' + s + '_' + r + '_' + q + '_p_' + p] = rnk
        else:
            if q != 'Q8':
                for j, (k, v) in enumerate(d.items()):
                    cdf['s' + s + '_' + q + '_p_' + k] = v

                    if q != 'Q6':# not ranking question
                        cdf['s' + s + '_' + q + '_p_' + k] = v/ 100
                        ### calculate irrationality for participant
                        if 'and' in k:
                            pix = set([0, 1, 2]) - set([j])  # indices of single probabilities.
                            pvals = np.array(list(d.values()), dtype='float') / 100. # the probabilities from the user.
                            cdf['s' + s + '_' + q + '_irr'] = pvals[j] - np.array(pvals)[list(pix)].min()

                if q == 'Q6':
                    cdf[cdf.columns[cdf.columns.str.contains('s' + s + '_' + q)]] = cdf[cdf.columns[cdf.columns.str.contains('s' + s + '_' + q)]].astype('int').rank(axis=1)

            else:
                cdf['s' + s + '_' + q + '_chosen_'] = d[0]

    return cdf


### find all csv files in folder
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))

stories = set(['suspect', 'art'])

# for j, f in enumerate(files):
for j, f in enumerate(files):
    ### reformated data for each user
    cdf = pd.DataFrame()
    print(f)
    df = pd.read_csv(f, index_col=0)
    df.reset_index(inplace=True, drop=True)
    d = ast.literal_eval(df.loc[df['state'] == 'robots_setup', 'val'].values[0])
    for k, v in d.items():
        cdf.loc[0, k] = v

    ### split the data set to 2 dataframes
    s, e = df[df['state'] == 'story0'].index
    d1 = df.iloc[s:e - 1, :].copy()
    d1.reset_index(inplace=True, drop=True)
    d2 = df.iloc[e:-1, :].copy()
    d2.reset_index(inplace=True, drop=True)
    dfs = [d1, d2]
    for i, df in enumerate(dfs):
        for q, ix in participant_questions.items():
            cdf = dfstr2dict(cdf, df, ix, q, i)
        for q, ix in robots_questions.items():
            cdf = dfstr2dict(cdf, df, ix, q, i)
        for q, ix in preference_questions.items():
            cdf['s' + str(i) + '_' + q] = df.loc[ix, 'val'].rstrip("\n")

    ### change columns names: color --> color_rationality
    newcols = list(cdf.columns[:7]) + list(cdf.columns[7:].str.replace('red', cdf['red rationality'].values[0]))
    cdf.columns = newcols
    newcols = list(cdf.columns[:7]) + list(cdf.columns[7:].str.replace('blue', cdf['blue rationality'].values[0]))
    cdf.columns = newcols

    ### change values: color --> color_rationality
    cdf.replace('red', cdf['red rationality'].values[0], inplace=True)
    cdf.replace('blue', cdf['blue rationality'].values[0], inplace=True)

    ### ranking preference
    for s in [0,1]:
        cdf['s%d_first_choice' %s] = cdf[cdf.columns[cdf.columns.str.contains('s%d_Q6' %s)]].astype('int').idxmin(1)[0].split('p_')[1]
        cdf['s%d_rational_first_choice' % s] = cdf[cdf.columns[cdf.columns.str.contains('s%d_rational_Q7' %s)]].astype('int').idxmin(1)[0].split('p_')[1]
        cdf['s%d_irrational_first_choice' % s] =cdf[cdf.columns[cdf.columns.str.contains('s%d_irrational_Q7' %s)]].astype('int').idxmin(1)[0].split('p_')[1]
        cdf['s%d_final_choice' %s] = cdf[cdf.columns[cdf.columns.str.contains('s%d_Q8' %s)]]
        cdf['s%d_consistent_choice' %s] = int(cdf['s%d_final_choice' %s] == cdf['s%d_first_choice' %s])
        ### is equal to one of the robots ...
        if (cdf['s%d_final_choice' %s].values == cdf['s%d_rational_first_choice' % s].values)[0]:
            cdf['s%d_choice_equal2' % s] = 'rational'
        elif (cdf['s%d_final_choice' %s]  == cdf['s%d_irrational_first_choice' % s])[0]:
            cdf['s%d_choice_equal2' % s] = 'irrational'
        else:
            cdf['s%d_choice_equal2' % s] = 0

        ### replace s0/1 by art/suspect for which robot you agree with
        fst_story = cdf['first story']
        scnd_story = list(stories - set(fst_story))
        sq1 = 's%d_Q2' % s
        sq5 = 's%d_Q5' % s
        cdf[sq1.replace('s%d' % s, fst_story.values[0])] = cdf['s%d_Q2' % s]
        cdf[sq5.replace('s%d' % s, fst_story.values[0])] = cdf['s%d_Q2' % s]

    # cdf['s%d_first_choice' % s] = cdf[cdf.columns[cdf.columns.str.contains('s%d_Q6' % s)]].astype('int').idxmin(1)[0]

    if 'all_users_df' in locals():
        all_users_df = all_users_df.append(cdf, ignore_index=True)
    else:
        all_users_df = cdf.copy()

all_users_df = all_users_df.iloc[6:,:]
all_users_df.to_csv('./data/reformatted_experiment_data/all_users_logs.csv', index_label=0)


qual_df = pd.read_csv('./data/qualtrics/pre_interaction.csv')
### preference questions
qual_df = qual_df.iloc[9:,:] ### filter first users that the program had a bug

### typo error
qual_df.loc[qual_df['ResponseId'] == 'R_9ZGrLH1dEw1lr8t', 'Q1'] = 30

### reformmating robot chices. 1: blue, 2: red
q15cols = list(qual_df.columns[qual_df.columns.str.contains('Q15')])
choices_cols = ['Q8', 'Q9','Q11','Q12', 'Q16'] + q15cols

qual_df[choices_cols] = qual_df[choices_cols].astype('int')
qual_df['Q11'].replace({2:1, 3:2}, inplace=True)
qual_df[q15cols] = qual_df[q15cols].replace({3:2})

### users that snwer manually due to rushing...
qual_df.loc[qual_df['Q1'] == '012', ['Q8', 'Q9','Q11','Q12']] = [2 ,1 ,1 ,1]
qual_df.loc[qual_df['Q1'] == '024', 'Q8'] = 2
qual_df.loc[qual_df['Q1'] == '030', 'Q8'] = 2

rationality_dict = {'rational' : 1, 'irrational': -1}

for id in all_users_df['user id'].unique():
    qual_df.loc[qual_df['Q1'] == id, choices_cols] = qual_df.loc[qual_df['Q1'] == id, choices_cols].replace({1: all_users_df.loc[all_users_df['user id'] == id, 'blue rationality'].replace(rationality_dict).values.flatten()[0],
                                                                                                             2: all_users_df.loc[all_users_df['user id'] == id, 'red rationality'].replace(rationality_dict).values.flatten()[0]})

### sort both dataframes by ID
qual_df['Q1'] = qual_df['Q1'].astype('int')
qual_df.sort_values(by='Q1')
all_users_df.sort_values(by='user id')

### reset indecies
qual_df.reset_index(drop=True, inplace=True)
all_users_df.reset_index(drop=True, inplace=True)

choices_app = list(all_users_df.columns[all_users_df.columns.str.contains('Q2')]) + list(all_users_df.columns[all_users_df.columns.str.contains('Q5')])
all_users_df[choices_app] = all_users_df[choices_app].replace(rationality_dict)

### story position to art/suspect


# cdf[] = cdf['s%d_Q2' % s]
cdf[sq5.replace('s%d' % s, fst_story.values[0])] = cdf['s%d_Q2' % s]
for row in all_users_df.iterrows():
    for i in [0,1]:
        fst_story = row[1]['first story']
        scnd_story = list(stories - set(fst_story))
        sq1 = 's%d_Q2' % i
        sq5 = 's%d_Q5' % i
        # todo: continue here...
        qual_df[sq1.replace('s%d' % i, fst_story.values[0])] = qual_df['s%d_Q2' % i]
        qual_df[sq5.replace('s%d' % i, fst_story.values[0])] = qual_df['s%d_Q2' % i]

chosen_robot_df = pd.concat((qual_df.loc[:, choices_cols], all_users_df[choices_app]), axis = 1)

### replace irrational value to 0 because it easier to work with it.
chosen_robot_df = chosen_robot_df.replace({-1:0})

###  rename the columns according to their meaning
chosen_robot_df = chosen_robot_df.rename(qualtrics_quetions, axis = 1)

### calculate how many times the rational robot was agreed with in th first/ second story and in total.
chosen_robot_df['s0_chose_rational'] = chosen_robot_df[sclmns(chosen_robot_df, ['s0'])].sum(axis=1) / 2 # first
chosen_robot_df['s1_chose_rational'] = chosen_robot_df[sclmns(chosen_robot_df, ['s1'])].sum(axis=1) / 2 # second
chosen_robot_df['tot_chose_rational'] = chosen_robot_df['s0_chose_rational'] / 2 + chosen_robot_df['s1_chose_rational'] / 2 # total

pvs, corr_df = calculate_corr_with_pvalues(chosen_robot_df)

corr_df[pvs<0.05] = 0

sns.heatmap(chosen_robot_df.corr()[pvs<.05], annot=True)
plt.show()

### which robot was chosen when.
chosen_robot_df.hist()
qual_df['Q3'].value_counts() # 1: man, 2: woman


all_users_df[sclmns(all_users_df, ['_first_choice', '_final_choice', '_consistent_choice'])]

print()
