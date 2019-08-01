from calc_explore_U import *

h_art     = [-1,0.3160876599,0.1537699041,0.1902346651,0.3837542279,-0.368968738,1,0.3286240466,-0.9634234453,0.063746448]
h_suspect = [-0.9997512685,-0.1999484744,1,0.2876677585,-0.9000930821,-0.1915275209,0.3275892904,0.4479089012,0.889189512,0.5750046338]

U_art     = U_from_H(grandH_from_x(h_art))
U_suspect = U_from_H(grandH_from_x(h_suspect))


pd.DataFrame(U_art).to_csv('U_art.csv')
pd.DataFrame(U_suspect).to_csv('U_suspect.csv')