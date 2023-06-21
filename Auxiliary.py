# Maksim Terentev
# Auxiliary functions
# Last changes: 21/06/2023
# Version 1.0.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import cm

# Contains the test responses of human participants and GPT models
def responses():
    # nd arrays: participants (15) or GPT model test runs (3) x question (36)
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
    return participants_raw, davinci_raw, gpt3_raw, gpt4_raw

# Summary statistics function
def summary_statistics():
    participants_raw, davinci_raw, gpt3_raw, gpt4_raw = responses()
    
    # Human Participants
    print("---------------------------------------------")
    print("Human Participants")
    print("mean: ", round(np.mean(np.sum(participants_raw, axis = 1)), 1))
    print("sd: ", round(np.std(np.sum(participants_raw, axis = 1)), 1))
    print("max: ", np.amax(np.sum(participants_raw, axis = 1)))
    print("min: ", np.amin(np.sum(participants_raw, axis = 1)))
    print("---------------------------------------------")
    
    # text-davinci-003
    print("text-davinci-003")
    print("mean: ", round(np.mean(np.sum(davinci_raw, axis = 1)), 1))
    print("sd: ", round(np.std(np.sum(davinci_raw, axis = 1)), 1))
    print("max: ", np.amax(np.sum(davinci_raw, axis = 1)))
    print("min: ", np.amin(np.sum(davinci_raw, axis = 1)))
    print("---------------------------------------------")
    
    # gpt-3.5-turbo
    print("gpt-3.5-turbo")
    print("mean: ", round(np.mean(np.sum(gpt3_raw, axis = 1)), 1))
    print("sd: ", round(np.std(np.sum(gpt3_raw, axis = 1)), 1))
    print("max: ", np.amax(np.sum(gpt3_raw, axis = 1)))
    print("min: ", np.amin(np.sum(gpt3_raw, axis = 1)))
    print("---------------------------------------------")
    
    # gpt-4
    print("gpt-4")
    print("mean: ", round(np.mean(np.sum(gpt4_raw, axis = 1)), 1))
    print("sd: ", round(np.std(np.sum(gpt4_raw, axis = 1)), 1))
    print("max: ", np.amax(np.sum(gpt4_raw, axis = 1)))
    print("min: ", np.amin(np.sum(gpt4_raw, axis = 1)))
    print("---------------------------------------------")
    
    participants_data = np.sum(participants_raw, axis = 1)
    davinci_data = np.sum(davinci_raw, axis = 1)
    gpt3_data = np.sum(gpt3_raw, axis = 1)
    gpt4_data = np.sum(gpt4_raw, axis = 1)
    
    fig = plt.figure(figsize = (8, 5))
    plt.ylabel('Total Score')
    plt.boxplot(x = [participants_data, davinci_data, gpt3_data, gpt4_data], labels = ["Human Participants", "text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
    
    x_1 = np.random.normal(1, 0.0, len(participants_data))
    x_2 = np.random.normal(2, 0.0, len(davinci_data))
    x_3 = np.random.normal(3, 0.0, len(gpt3_data))
    x_4 = np.random.normal(4, 0.0, len(gpt4_data))
    
    # Scatter points
    plt.scatter(x_1, participants_data, color = 'blue', alpha = 0.6)
    plt.scatter(x_2, davinci_data, color = 'blue', alpha = 0.6)
    plt.scatter(x_3, gpt3_data, color = 'blue', alpha = 0.6)
    plt.scatter(x_4, gpt4_data, color = 'blue', alpha = 0.6)

    plt.yticks([20,22,24,26,28,30,32,34,36])
    plt.show()
     
# Preprocess the responses for the performance per test type plot   
def preprocess_ToM_stories():
    participants_raw, davinci_raw, gpt3_raw, gpt4_raw = responses()
    # Per model, for each question, take majority as the result (mode)
    participants = stats.mode(participants_raw, keepdims = True)[0].flatten()
    davinci = stats.mode(davinci_raw, keepdims = True)[0].flatten()
    gpt3 = stats.mode(gpt3_raw, keepdims = True)[0].flatten()
    gpt4 =  stats.mode(gpt4_raw, keepdims = True)[0].flatten()
    # Regroup stories per category
    UTT = np.array([[participants[0], participants[1], participants[2], participants[3], participants[4], participants[5], participants[6], participants[7], participants[8]],
                    [davinci[0], davinci[1], davinci[2], davinci[3], davinci[4], davinci[5], davinci[6], davinci[7], davinci[8]],
                    [gpt3[0], gpt3[1], gpt3[2], gpt3[3], gpt3[4], gpt3[5], gpt3[6], gpt3[7], gpt3[8]],
                    [gpt4[0], gpt4[1], gpt4[2], gpt4[3], gpt4[4], gpt4[5], gpt4[6], gpt4[7], gpt4[8]]])
    UCT = np.array([[participants[27], participants[28], participants[29], participants[30], participants[31], participants[32], participants[33], participants[34], participants[35]],
                    [davinci[27], davinci[28], davinci[29], davinci[30], davinci[31], davinci[32], davinci[33], davinci[34], davinci[35]],
                    [gpt3[27], gpt3[28], gpt3[29], gpt3[30], gpt3[31], gpt3[32], gpt3[33], gpt3[34], gpt3[35]],
                    [gpt4[27], gpt4[28], gpt4[29], gpt4[30], gpt4[31], gpt4[32], gpt4[33], gpt4[34], gpt4[35]]])
    ICTT = np.array([[participants[15], participants[16], participants[17], participants[18], participants[19], participants[20]],
                     [davinci[15], davinci[16], davinci[17], davinci[18], davinci[19], davinci[20]],
                     [gpt3[15], gpt3[16], gpt3[17], gpt3[18], gpt3[19], gpt3[20]],
                     [gpt4[15], gpt4[16], gpt4[17], gpt4[18], gpt4[19], gpt4[20]]])
    PT = np.array([[participants[9], participants[10], participants[11], participants[12], participants[13], participants[14]],
                   [davinci[9], davinci[10], davinci[11], davinci[12], davinci[13], davinci[14]],
                   [gpt3[9], gpt3[10], gpt3[11], gpt3[12], gpt3[13], gpt3[14]],
                   [gpt4[9], gpt4[10], gpt4[11], gpt4[12], gpt4[13], gpt4[14]]])
    LT = np.array([[participants[21], participants[22], participants[23]],
                   [davinci[21], davinci[22], davinci[23]],
                   [gpt3[21], gpt3[22], gpt3[23]],
                   [gpt4[21], gpt4[22], gpt4[23]]])
    WLT = np.array([[participants[24], participants[25], participants[26]],
                   [davinci[24], davinci[25], davinci[26]],
                   [gpt3[24], gpt3[25], gpt3[26]],
                   [gpt4[24], gpt4[25], gpt4[26]]])
    return UTT, UCT, ICTT, PT, LT, WLT
    
# Create the performance plot based on the type of the ToM story
def performance_per_story_type():
    UTT, UCT, ICTT, PT, LT, WLT = preprocess_ToM_stories()
    categories = ["Unexpected Transfer", "Unexpected Content", "Ice Cream Truck", "Prank", "Lie", "White Lie"]
    
    participants_results = [np.sum(UTT[0, :]) / 9 * 100, np.sum(UCT[0, :]) / 9 * 100, np.sum(ICTT[0, :]) / 6 * 100, np.sum(PT[0, :]) / 6 * 100, np.sum(LT[0, :]) / 3 * 100, np.sum(WLT[0, :]) / 3 * 100]
    davinci_results = [np.sum(UTT[1, :]) / 9 * 100, np.sum(UCT[1, :]) / 9 * 100, np.sum(ICTT[1, :]) / 6 * 100, np.sum(PT[1, :]) / 6 * 100, np.sum(LT[1, :]) / 3 * 100, np.sum(WLT[1, :]) / 3 * 100]
    gpt3_results = [np.sum(UTT[2, :]) / 9 * 100, np.sum(UCT[2, :]) / 9 * 100, np.sum(ICTT[2, :]) / 6 * 100, np.sum(PT[2, :]) / 6 * 100, np.sum(LT[2, :]) / 3 * 100, np.sum(WLT[2, :]) / 3 * 100]
    gpt4_results = [np.sum(UTT[3, :]) / 9 * 100, np.sum(UCT[3, :]) / 9 * 100, np.sum(ICTT[3, :]) / 6 * 100, np.sum(PT[3, :]) / 6 * 100, np.sum(LT[3, :]) / 3 * 100, np.sum(WLT[3, :]) / 3 * 100]
    
    df = pd.DataFrame({ "Participants" : participants_results, "text-davinci-003" : davinci_results, "gpt-3.5-turbo" : gpt3_results, "gpt-4" : gpt4_results}, index = categories)
    color = cm.inferno_r(np.linspace(.9, .2, 4))
    ax = df.plot.barh(figsize = (13, 7), width = 0.6, color = color, edgecolor = 'black', linewidth = 0.2)
    ax.set_xlabel("Passing Percentage")
    ax.set_title("Performance on ToM tests")
    ax.xaxis.grid(True, color = "#DFDFDF")
    plt.xlim([0, 101])
    plt.show()
    
   # Preprocess the responses for the performance per questions type plot        
def preprocess_ToM_questions():
    participants_raw, davinci_raw, gpt3_raw, gpt4_raw = responses()
    # Per model, for each question, take majority as the result (mode)
    participants = stats.mode(participants_raw, keepdims = True)[0].flatten()
    davinci = stats.mode(davinci_raw, keepdims = True)[0].flatten()
    gpt3 = stats.mode(gpt3_raw, keepdims = True)[0].flatten()
    gpt4 =  stats.mode(gpt4_raw, keepdims = True)[0].flatten()
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

# Create the performance plot based on the type of the ToM question
def performance_per_question_type():
    reality, first_order, second_order, third_order = preprocess_ToM_questions()
        
    categories = ["Reality", "First-order", "Second-order", "Third-order"]
    
    participants_results = [np.sum(reality[0, :]) / 13 * 100, np.sum(first_order[0, :]) / 10 * 100, np.sum(second_order[0, :]) / 11 * 100, np.sum(third_order[0, :]) / 2 * 100]
    davinci_results = [np.sum(reality[1, :]) / 13 * 100, np.sum(first_order[1, :]) / 10 * 100, np.sum(second_order[1, :]) / 11 * 100, np.sum(third_order[1, :]) / 2 * 100]
    gpt3_results = [np.sum(reality[2, :]) / 13 * 100, np.sum(first_order[2, :]) / 10 * 100, np.sum(second_order[2, :]) / 11 * 100, np.sum(third_order[2, :]) / 2 * 100]
    gpt4_results = [np.sum(reality[3, :]) / 13 * 100, np.sum(first_order[3, :]) / 10 * 100, np.sum(second_order[3, :]) / 11 * 100, np.sum(third_order[3, :]) / 2 * 100]
    
    df = pd.DataFrame({ "Participants" : participants_results, "text-davinci-003" : davinci_results, "gpt-3.5-turbo" : gpt3_results, "gpt-4" : gpt4_results}, index = categories)
    color = cm.viridis_r(np.linspace(.9, .2, 4))
    ax = df.plot.barh(figsize = (11, 7), width = 0.6, color = color, edgecolor = 'black', linewidth = 0.2)
    ax.set_xlabel("Passing Percentage")
    ax.set_title("Performance on ToM tests")
    ax.xaxis.grid(True, color = "#DFDFDF")
    plt.xlim([0, 101])
    plt.show()