import numpy as np
import pandas as pd

all_data = np.load('data/processed_data/all_data_dict_gamma.npy').item()

err_df = pd.DataFrame(columns=['id', 'e_a', 'e_b', 'e_ab'])
for i, uid in enumerate(all_data.keys()):
    err_df.loc[i, 'id'] = uid
    err_df.loc[i, ['e_a', 'e_b', 'e_ab']] = all_data[uid]['pred_errors']

err_df = err_df.astype('float')
err_df.to_csv('data/new_code/prediction_errors_gamma.csv')  # index=False)

print(err_df.describe())