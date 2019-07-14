# Check our quantum model.
For further information go to project --> quantum_model.


Code needed for pre-processing the data for the irrational robots experiment.

raw data from qualtrics: /data/raw_data/calc_u_suspects - only the 1 (third) question.
<!--raw data from qualtrics: /data/raw_data/amt4u_calc - this 4 questions (third) randomized/ ~100 participants (total)-->

Order of files --> follow instructions in each file.

1. reformat_data.py:  organize the data. --> /data/processed_data
2. calc_explore_U.py: calculate {h_i}, U and predict {p_i} and check BIC, RMSE. --> /data/predictions

3. supporting files: general_quantum_operators, hamiltonian_prediction, minimization_functions.