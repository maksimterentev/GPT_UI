# Maksim Terentev
# ToM Experiments
# Last changes: 18/06/2023
# Version 1.0.0

import numpy as np
from scipy import stats


def preprocess():
    # nd arrays: participants (15) or model test runs (3) x question (36)
    # Questions order: FBT_1.1, FBT_1.2, FBT_1.3, FBT_2.1, etc.
    # 0 - incorrect; 1 - correct.
    participants_raw = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1],
                            [1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,1,0,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,1,0,1,0,0,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    davinci_raw = np.array([[1,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0],
                       [1,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0],
                       [1,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0]])
    gpt3_raw = np.array([[1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,1],
                    [1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,1],
                    [1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,1]])
    gpt4_raw = np.array([[1,0,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
                    [1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
                    [1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1]])
    # Per model, for each question, take majority as the result (mode)
    participants = stats.mode(participants_raw, keepdims = True)[0].flatten()
    davinci = stats.mode(davinci_raw, keepdims = True)[0].flatten()
    gpt3 = stats.mode(gpt3_raw, keepdims = True)[0].flatten()
    gpt4 =  stats.mode(gpt4_raw, keepdims = True)[0].flatten()
    # Regroup questions per order: reality questions, first order, second order, third order
    # Each row - the model, (participants/davinci/gpt3/gpt4) each column, the question
    # Reality questions: FBT_1.1,FBT_2.1,FBT_3.1, FBT_3.2, FBT_4.1, FBT_6.1, FBT_7.1, FBT_8.1, FBT_9.1, FBT_10.1, FBT_11.2, FBT_11.3, FBT_12.2
    reality = np.array([[participants[0], participants[3], participants[6], participants[7], participants[9], participants[15], participants[18], participants[21], participants[24], participants[27], participants[31], participants[32], participants[34]],
                                  [davinci[0], davinci[3], davinci[6], davinci[7], davinci[9], davinci[15], davinci[18], davinci[21], davinci[24], davinci[27], davinci[31], davinci[32], davinci[34]],
                                  [gpt3[0], gpt3[3], gpt3[6], gpt3[7], gpt3[9], gpt3[15], gpt3[18], gpt3[21], gpt3[24], gpt3[27], gpt3[31], gpt3[32], gpt3[34]],
                                  [gpt4[0], gpt4[3], gpt4[6], gpt4[7], gpt4[9], gpt4[15], gpt4[18], gpt4[21], gpt4[24], gpt4[27], gpt4[31], gpt4[32], gpt4[34]]])
    
    # First order false-belief quiestions: FBT_1.2, FBT_2.2, FBT_3.3, FBT_4.2, FBT_5.2, FBT_8.2, FBT_9.2, FBT_11.1, FBT_12.1, FBT_12.3
    first_order = np.array([[participants[1], participants[4], participants[8], participants[10], participants[13], participants[22], participants[25], participants[30], participants[33], participants[35]],
                            [davinci[1], davinci[4], davinci[8], davinci[10], davinci[13], davinci[22], davinci[25], davinci[30], davinci[33], davinci[35]],
                            [gpt3[1], gpt3[4], gpt3[8], gpt3[10], gpt3[13], gpt3[22], gpt3[25], gpt3[30], gpt3[33], gpt3[35]],
                            [gpt4[1], gpt4[4], gpt4[8], gpt4[10], gpt4[13], gpt4[22], gpt4[25], gpt4[30], gpt4[33], gpt4[35]]])
    # Second order false-belief questions: FBT_1.3, FBT_2.3, FBT_5.1, FBT_6.2, FBT_6.3, FBT_7.2, FBT_7.3, FBT_8.3, FBT_9.3, FBT_10.2, FBT_10.3
    second_order = np.array([[participants[2], participants[5], participants[12], participants[16], participants[17], participants[19], participants[20], participants[23], participants[26], participants[28], participants[29]],
                            [davinci[2], davinci[5], davinci[12], davinci[16], davinci[17], davinci[19], davinci[20], davinci[23], davinci[26], davinci[28], davinci[29]],
                            [gpt3[2], gpt3[5], gpt3[12], gpt3[16], gpt3[17], gpt3[19], gpt3[20], gpt3[23], gpt3[26], gpt3[28], gpt3[29]],
                            [gpt4[2], gpt4[5], gpt4[12], gpt4[16], gpt4[17], gpt4[19], gpt4[20], gpt4[23], gpt4[26], gpt4[28], gpt4[29]]])
    # Third order false-belief questions: FBT_4.3, FBT_5.3 
    third_order = np.array([[participants[11], participants[14]],
                            [davinci[11], davinci[14]],
                            [gpt3[11], gpt3[14]],
                            [gpt4[11], gpt4[14]]])    
    return reality, first_order, second_order, third_order
    
    def stat_test():
        #contingency_table = np.array([[np.sum((human == 1) & (gpt4 == 1)), np.sum((human == 1) & (gpt4 == 0))],
        #                          [np.sum((human == 0) & (gpt4 == 1)), np.sum((human == 0) & (gpt4 == 0))]])
        #print(contingency_table)
        #chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    
        # Cannot reject h_0: there is no significant difference
        pass
