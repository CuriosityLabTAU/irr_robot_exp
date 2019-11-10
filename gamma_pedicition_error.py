import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

all_data = np.load('data/processed_data/all_data_dict_gamma.npy', allow_pickle=True).item()

err_df = pd.DataFrame(columns=['id', 'e_a_pred', 'e_d_pred', 'e_ad_pred',
                               'e_a_pre', 'e_d_pre',
                               'e_a_mean_pre', 'e_d_mean_pre',
                               'e_a_mean3', 'e_d_mean3', 'e_ad_mean3'])
for i, uid in enumerate(all_data.keys()):
    err_df.loc[i, 'id'] = uid
    err_df.loc[i, ['e_a_pred', 'e_d_pred', 'e_ad_pred']] = all_data[uid]['pred_errors']
    err_df.loc[i, ['e_a_pre', 'e_d_pre']] = all_data[uid]['pre_errors']
    err_df.loc[i, ['e_a_mean_pre', 'e_d_mean_pre']] = all_data[uid]['mean_pre_err']
    err_df.loc[i, ['e_a_mean3', 'e_d_mean3', 'e_ad_mean3']] = all_data[uid]['mean_3_err']

err_df = err_df.astype('float')
err_df.to_csv('data/new_code/prediction_errors_gamma.csv')  # index=False)


fig, ax = plt.subplots(1,1)
sns.boxplot(data=err_df[['e_a_pred', 'e_d_pred', 'e_ad_pred',
                               'e_a_pre', 'e_d_pre',
                               'e_a_mean_pre', 'e_d_mean_pre',
                               'e_a_mean3', 'e_d_mean3', 'e_ad_mean3']], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
ax.set_ylabel('Error (RMSE)')
fig.savefig('errors_RMSE_boxplot.png', dpi=300)

print(err_df.describe())
plt.show()