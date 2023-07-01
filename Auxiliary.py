# Maksim Terentev
# Auxiliary functions
# Last changes: 24/06/2023
# Version 1.2

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import cm

# Contains the ToM tests responses of human participants and GPT models
def responses():
    # np arrays: participants (15) OR GPT models test runs (3) x question (36)
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
    gpt_3_5_raw = np.array([[1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,1],
                    [1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,1],
                    [1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,1]])
    gpt_4_raw = np.array([[1,0,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
                    [1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1],
                    [1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1]])
    return participants_raw, davinci_raw, gpt_3_5_raw, gpt_4_raw

# The summary statistics function
def summary_statistics():
    participants_raw, text_davinci_003_raw, gpt_3_5_raw, gpt_4_raw = responses()
    
    # Human Participants
    print("---------------------------------------------")
    print("Human Participants")
    print("Max: ", np.amax(np.sum(participants_raw, axis = 1)))
    print("Min: ", np.amin(np.sum(participants_raw, axis = 1)))
    print("Mean: ", round(np.mean(np.sum(participants_raw, axis = 1)), 1))
    print("SD: ", round(np.std(np.sum(participants_raw, axis = 1)), 1))
    print("---------------------------------------------")
    
    # text-davinci-003
    print("text-davinci-003")
    print("Max: ", np.amax(np.sum(text_davinci_003_raw, axis = 1)))
    print("Min: ", np.amin(np.sum(text_davinci_003_raw, axis = 1)))
    print("Mean: ", round(np.mean(np.sum(text_davinci_003_raw, axis = 1)), 1))
    print("SD: ", round(np.std(np.sum(text_davinci_003_raw, axis = 1)), 1))
    print("---------------------------------------------")
    
    # gpt-3.5-turbo
    print("gpt-3.5-turbo")
    print("Max: ", np.amax(np.sum(gpt_3_5_raw, axis = 1)))
    print("Min: ", np.amin(np.sum(gpt_3_5_raw, axis = 1)))
    print("Mean: ", round(np.mean(np.sum(gpt_3_5_raw, axis = 1)), 1))
    print("SD: ", round(np.std(np.sum(gpt_3_5_raw, axis = 1)), 1))
    print("---------------------------------------------")
    
    # gpt-4
    print("gpt-4")
    print("Max: ", np.amax(np.sum(gpt_4_raw, axis = 1)))
    print("Min: ", np.amin(np.sum(gpt_4_raw, axis = 1)))
    print("Mean: ", round(np.mean(np.sum(gpt_4_raw, axis = 1)), 1))
    print("SD: ", round(np.std(np.sum(gpt_4_raw, axis = 1)), 1))
    print("---------------------------------------------")
    
    # SS plot
    participants_data = np.sum(participants_raw, axis = 1)
    text_davinci_003_data = np.mean(np.sum(text_davinci_003_raw, axis = 1))
    gpt_3_5_data = np.mean(np.sum(gpt_3_5_raw, axis = 1))
    gpt_4_data = np.mean(np.sum(gpt_4_raw, axis = 1))
    
    fig, ax = plt.subplots(figsize = (7, 7))
    ax.boxplot(participants_data, positions = [1], widths = 0.4)
    
    ax.scatter(np.full_like(participants_data, 1), participants_data, color = 'blue', label = 'Human Participants')
    ax.scatter(np.full_like(text_davinci_003_data, 2), text_davinci_003_data, color = 'blue', label = 'text-davinci-003')
    ax.scatter(np.full_like(gpt_3_5_data, 3), gpt_3_5_data, color = 'blue', label = 'gpt-3.5-turbo')
    ax.scatter(np.full_like(gpt_4_data, 4), gpt_4_data, color = 'blue', label = 'gpt-4')
    
    ax.axhline(y = text_davinci_003_data, color = 'grey', linestyle = '--', alpha = 0.1, linewidth = 0.7)
    ax.axhline(y = gpt_3_5_data, color = 'grey', linestyle = '--', alpha = 0.1, linewidth = 0.7)
    ax.axhline(y = gpt_4_data, color = 'grey', linestyle = '--', alpha = 0.1, linewidth = 0.7)
    
    ax.axvline(x = 1, color = 'grey', linestyle = '--', alpha = 0.1, linewidth = 0.7)
    ax.axvline(x = 2, color = 'grey', linestyle = '--', alpha = 0.1, linewidth = 0.7)
    ax.axvline(x = 3, color = 'grey', linestyle = '--', alpha = 0.1, linewidth = 0.7)
    ax.axvline(x = 4, color = 'grey', linestyle = '--', alpha = 0.1, linewidth = 0.7)
    
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Human Participants', 'text-davinci-003', 'gpt-3.5-turbo', 'gpt-4'])
    ax.set_ylabel('Total Score')
    ax.set_ylim([0, 36])
    
    plt.show()
    #plt.savefig('SS_plot.png', dpi = 300)
     
# Preprocess the responses for the performance per the story-type plot
def preprocess_ToM_stories():
    participants_raw, text_davinci_003_raw, gpt_3_5_raw, gpt_4_raw = responses()
    # Per model, for each question, take the majority as a result (mode)
    participants = stats.mode(participants_raw, keepdims = True)[0].flatten()
    text_davinci_003 = stats.mode(text_davinci_003_raw, keepdims = True)[0].flatten()
    gpt_3_5 = stats.mode(gpt_3_5_raw, keepdims = True)[0].flatten()
    gpt_4 =  stats.mode(gpt_4_raw, keepdims = True)[0].flatten()
    # Regroup stories per category
    UTT = np.array([[participants[0], participants[1], participants[2], participants[3], participants[4], participants[5], participants[6], participants[7], participants[8], participants[15], participants[16], participants[17], participants[18], participants[19], participants[20]],
                    [text_davinci_003[0], text_davinci_003[1], text_davinci_003[2], text_davinci_003[3], text_davinci_003[4], text_davinci_003[5], text_davinci_003[6], text_davinci_003[7], text_davinci_003[8], text_davinci_003[15], text_davinci_003[16], text_davinci_003[17], text_davinci_003[18], text_davinci_003[19], text_davinci_003[20]],
                    [gpt_3_5[0], gpt_3_5[1], gpt_3_5[2], gpt_3_5[3], gpt_3_5[4], gpt_3_5[5], gpt_3_5[6], gpt_3_5[7], gpt_3_5[8], gpt_3_5[15], gpt_3_5[16], gpt_3_5[17], gpt_3_5[18], gpt_3_5[19], gpt_3_5[20]],
                    [gpt_4[0], gpt_4[1], gpt_4[2], gpt_4[3], gpt_4[4], gpt_4[5], gpt_4[6], gpt_4[7], gpt_4[8], gpt_4[15], gpt_4[16], gpt_4[17], gpt_4[18], gpt_4[19], gpt_4[20]]])
    UCT = np.array([[participants[27], participants[28], participants[29], participants[30], participants[31], participants[32], participants[33], participants[34], participants[35]],
                    [text_davinci_003[27], text_davinci_003[28], text_davinci_003[29], text_davinci_003[30], text_davinci_003[31], text_davinci_003[32], text_davinci_003[33], text_davinci_003[34], text_davinci_003[35]],
                    [gpt_3_5[27], gpt_3_5[28], gpt_3_5[29], gpt_3_5[30], gpt_3_5[31], gpt_3_5[32], gpt_3_5[33], gpt_3_5[34], gpt_3_5[35]],
                    [gpt_4[27], gpt_4[28], gpt_4[29], gpt_4[30], gpt_4[31], gpt_4[32], gpt_4[33], gpt_4[34], gpt_4[35]]])
    DBT = np.array([[participants[9], participants[10], participants[11], participants[12], participants[13], participants[14], participants[21], participants[22], participants[23], participants[24], participants[25], participants[26]],
                    [text_davinci_003[9], text_davinci_003[10], text_davinci_003[11], text_davinci_003[12], text_davinci_003[13], text_davinci_003[14], text_davinci_003[21], text_davinci_003[22], text_davinci_003[23], text_davinci_003[24], text_davinci_003[25], text_davinci_003[26]],
                    [gpt_3_5[9], gpt_3_5[10], gpt_3_5[11], gpt_3_5[12], gpt_3_5[13], gpt_3_5[14], gpt_3_5[21], gpt_3_5[22], gpt_3_5[23], gpt_3_5[24], gpt_3_5[25], gpt_3_5[26]],
                    [gpt_4[9], gpt_4[10], gpt_4[11], gpt_4[12], gpt_4[13], gpt_4[14], gpt_4[21], gpt_4[22], gpt_4[23], gpt_4[24], gpt_4[25], gpt_4[26]]])
    return UTT, UCT, DBT

# Generate the performance plot based on the type of the ToM story
def performance_per_story_type():
    UTT, UCT, DBT = preprocess_ToM_stories()
    categories = ["Unexpected Transfer", "Unexpected Content", "Deception-Based"]
    
    participants_results = [np.sum(UTT[0, :]) / 15 * 100, np.sum(UCT[0, :]) / 9 * 100, np.sum(DBT[0, :]) / 12 * 100]
    text_davinci_003_results = [np.sum(UTT[1, :]) / 15 * 100, np.sum(UCT[1, :]) / 9 * 100, np.sum(DBT[1, :]) / 12 * 100]
    gpt_3_5_results = [np.sum(UTT[2, :]) / 15 * 100, np.sum(UCT[2, :]) / 9 * 100, np.sum(DBT[2, :]) / 12 * 100]
    gpt_4_results = [np.sum(UTT[3, :]) / 15 * 100, np.sum(UCT[3, :]) / 9 * 100, np.sum(DBT[3, :]) / 12 * 100]
    
    df = pd.DataFrame({ "Participants" : participants_results, "text-davinci-003" : text_davinci_003_results, "gpt-3.5-turbo" : gpt_3_5_results, "gpt-4" : gpt_4_results}, index = categories)
    #color = cm.rainbow_r(np.linspace(0, 1, 4))
    color = {"Participants" : "red", "text-davinci-003" : "blue", "gpt-3.5-turbo" : "green", "gpt-4" : "orange"}
    ax = df.plot.barh(figsize = (13, 7), width = 0.6, color = color, edgecolor = 'black', linewidth = 0.2)
    ax.set_xlabel("Passing Rate")
    ax.set_title("Performance on ToM Tests per Story Type")
    ax.xaxis.grid(True, color = "#DFDFDF", alpha = 1)
    plt.xlim([0, 101])
    plt.show()
    #plt.savefig('PpS_plot.png', dpi = 300)

# Generate the performance plot based on the type of the ToM story per model
def performance_per_story_type_subplots():
    UTT, UCT, DBT = preprocess_ToM_stories()
    categories = ["Unexpected\nTransfer", "Unexpected\nContent", "Deception-\nBased"]
    
    participants_results = [np.sum(UTT[0, :]) / 15 * 100, np.sum(UCT[0, :]) / 9 * 100, np.sum(DBT[0, :]) / 12 * 100]
    text_davinci_003_results = [np.sum(UTT[1, :]) / 15 * 100, np.sum(UCT[1, :]) / 9 * 100, np.sum(DBT[1, :]) / 12 * 100]
    gpt_3_5_results = [np.sum(UTT[2, :]) / 15 * 100, np.sum(UCT[2, :]) / 9 * 100, np.sum(DBT[2, :]) / 12 * 100]
    gpt_4_results = [np.sum(UTT[3, :]) / 15 * 100, np.sum(UCT[3, :]) / 9 * 100, np.sum(DBT[3, :]) / 12 * 100]
   
    fig, ax = plt.subplots(2, 2, figsize = (8, 6))

    ax[0, 0].bar(categories, participants_results, width = 0.4, color = ["red", "green", "blue"])
    ax[0, 1].bar(categories, gpt_4_results, width = 0.4, color = ["red", "green", "blue"])
    ax[1, 0].bar(categories, gpt_3_5_results, width = 0.4, color = ["red", "green", "blue"])
    ax[1, 1].bar(categories, text_davinci_003_results, width = 0.4, color = ["red", "green", "blue"])

    ax[0, 0].set_title('Human Participants')
    ax[0, 0].set_ylabel('Passing Rate')
    ax[0, 1].set_title('gpt-4')
    ax[0, 1].set_ylabel('Passing Rate')
    ax[1, 0].set_title('gpt-3.5-turbo')
    ax[1, 0].set_ylabel('Passing Rate')
    ax[1, 1].set_title('text-davinci-003')
    ax[1, 1].set_ylabel('Passing Rate')
    
    for ax in ax.flat:
        ax.set_ylim(0, 105)

    plt.subplots_adjust(hspace = 0.4, wspace = 0.3)

    plt.show()
    #plt.savefig('PpT_subplots.png', dpi = 300)
    
# Preprocess the responses for the performance per the question-type plot    
def preprocess_ToM_questions():
    participants_raw, text_davinci_003_raw, gpt_3_5_raw, gpt_4_raw = responses()
    # Per model, for each question, take the majority as a result (mode)
    participants = stats.mode(participants_raw, keepdims = True)[0].flatten()
    text_davinci_003 = stats.mode(text_davinci_003_raw, keepdims = True)[0].flatten()
    gpt_3_5 = stats.mode(gpt_3_5_raw, keepdims = True)[0].flatten()
    gpt_4 =  stats.mode(gpt_4_raw, keepdims = True)[0].flatten()
    # Regroup stories per category
    reality = np.array([[participants[0], participants[3], participants[6], participants[7], participants[9], participants[15], participants[18], participants[21], participants[24], participants[27], participants[31], participants[32], participants[34]],
                                  [text_davinci_003[0], text_davinci_003[3], text_davinci_003[6], text_davinci_003[7], text_davinci_003[9], text_davinci_003[15], text_davinci_003[18], text_davinci_003[21], text_davinci_003[24], text_davinci_003[27], text_davinci_003[31], text_davinci_003[32], text_davinci_003[34]],
                                  [gpt_3_5[0], gpt_3_5[3], gpt_3_5[6], gpt_3_5[7], gpt_3_5[9], gpt_3_5[15], gpt_3_5[18], gpt_3_5[21], gpt_3_5[24], gpt_3_5[27], gpt_3_5[31], gpt_3_5[32], gpt_3_5[34]],
                                  [gpt_4[0], gpt_4[3], gpt_4[6], gpt_4[7], gpt_4[9], gpt_4[15], gpt_4[18], gpt_4[21], gpt_4[24], gpt_4[27], gpt_4[31], gpt_4[32], gpt_4[34]]])
    first_order = np.array([[participants[1], participants[4], participants[8], participants[10], participants[13], participants[22], participants[25], participants[30], participants[33], participants[35]],
                            [text_davinci_003[1], text_davinci_003[4], text_davinci_003[8], text_davinci_003[10], text_davinci_003[13], text_davinci_003[22], text_davinci_003[25], text_davinci_003[30], text_davinci_003[33], text_davinci_003[35]],
                            [gpt_3_5[1], gpt_3_5[4], gpt_3_5[8], gpt_3_5[10], gpt_3_5[13], gpt_3_5[22], gpt_3_5[25], gpt_3_5[30], gpt_3_5[33], gpt_3_5[35]],
                            [gpt_4[1], gpt_4[4], gpt_4[8], gpt_4[10], gpt_4[13], gpt_4[22], gpt_4[25], gpt_4[30], gpt_4[33], gpt_4[35]]])
    second_order = np.array([[participants[2], participants[5], participants[12], participants[16], participants[17], participants[19], participants[20], participants[23], participants[26], participants[28], participants[29]],
                            [text_davinci_003[2], text_davinci_003[5], text_davinci_003[12], text_davinci_003[16], text_davinci_003[17], text_davinci_003[19], text_davinci_003[20], text_davinci_003[23], text_davinci_003[26], text_davinci_003[28], text_davinci_003[29]],
                            [gpt_3_5[2], gpt_3_5[5], gpt_3_5[12], gpt_3_5[16], gpt_3_5[17], gpt_3_5[19], gpt_3_5[20], gpt_3_5[23], gpt_3_5[26], gpt_3_5[28], gpt_3_5[29]],
                            [gpt_4[2], gpt_4[5], gpt_4[12], gpt_4[16], gpt_4[17], gpt_4[19], gpt_4[20], gpt_4[23], gpt_4[26], gpt_4[28], gpt_4[29]]])
    third_order = np.array([[participants[11], participants[14]],
                            [text_davinci_003[11], text_davinci_003[14]],
                            [gpt_3_5[11], gpt_3_5[14]],
                            [gpt_4[11], gpt_4[14]]])  
    return reality, first_order, second_order, third_order

# Generate the performance plot based on the type of the ToM question
def performance_per_question_type():
    reality, first_order, second_order, third_order = preprocess_ToM_questions()
        
    categories = ["Reality", "First-order", "Second-order", "Third-order"]
    
    participants_results = [np.sum(reality[0, :]) / 13 * 100, np.sum(first_order[0, :]) / 10 * 100, np.sum(second_order[0, :]) / 11 * 100, np.sum(third_order[0, :]) / 2 * 100]
    text_davinci_003_results = [np.sum(reality[1, :]) / 13 * 100, np.sum(first_order[1, :]) / 10 * 100, np.sum(second_order[1, :]) / 11 * 100, np.sum(third_order[1, :]) / 2 * 100]
    gpt_3_5_results = [np.sum(reality[2, :]) / 13 * 100, np.sum(first_order[2, :]) / 10 * 100, np.sum(second_order[2, :]) / 11 * 100, np.sum(third_order[2, :]) / 2 * 100]
    gpt_4_results = [np.sum(reality[3, :]) / 13 * 100, np.sum(first_order[3, :]) / 10 * 100, np.sum(second_order[3, :]) / 11 * 100, np.sum(third_order[3, :]) / 2 * 100]
    
    df = pd.DataFrame({ "Participants" : participants_results, "text-davinci-003" : text_davinci_003_results, "gpt-3.5-turbo" : gpt_3_5_results, "gpt-4" : gpt_4_results}, index = categories)
    #color = cm.viridis_r(np.linspace(.9, .2, 4))
    color = {"Participants" : "red", "text-davinci-003" : "blue", "gpt-3.5-turbo" : "green", "gpt-4" : "orange"}
    ax = df.plot.barh(figsize = (11, 7), width = 0.7, color = color, edgecolor = 'black', linewidth = 0.2)
    ax.set_xlabel("Passing Rate")
    ax.set_title("Performance on ToM Tests per Question Type")
    ax.xaxis.grid(True, color = "#DFDFDF", alpha = 1)
    plt.xlim([0, 101])
    plt.show()
    #plt.savefig('PpQ_plot.png', dpi = 300)
    
# Generate the performance plot based on the type of the ToM question per model
def performance_per_question_type_subplots():
   reality, first_order, second_order, third_order = preprocess_ToM_questions()
   
   categories = ["Reality", "First-\norder", "Second-\norder", "Third-\norder"]
   
   participants_results = [np.sum(reality[0, :]) / 13 * 100, np.sum(first_order[0, :]) / 10 * 100, np.sum(second_order[0, :]) / 11 * 100, np.sum(third_order[0, :]) / 2 * 100]
   text_davinci_003_results = [np.sum(reality[1, :]) / 13 * 100, np.sum(first_order[1, :]) / 10 * 100, np.sum(second_order[1, :]) / 11 * 100, np.sum(third_order[1, :]) / 2 * 100]
   gpt_3_5_results = [np.sum(reality[2, :]) / 13 * 100, np.sum(first_order[2, :]) / 10 * 100, np.sum(second_order[2, :]) / 11 * 100, np.sum(third_order[2, :]) / 2 * 100]
   gpt_4_results = [np.sum(reality[3, :]) / 13 * 100, np.sum(first_order[3, :]) / 10 * 100, np.sum(second_order[3, :]) / 11 * 100, np.sum(third_order[3, :]) / 2 * 100]
   
   fig, ax = plt.subplots(2, 2, figsize = (8, 6))
   ax[0, 0].bar(categories, participants_results, width = 0.5, color = ["red", "green", "blue", "purple"])
   ax[0, 1].bar(categories, gpt_4_results, width = 0.5, color = ["red", "green", "blue", "purple"])
   ax[1, 0].bar(categories, gpt_3_5_results, width = 0.5, color = ["red", "green", "blue", "purple"])
   ax[1, 1].bar(categories, text_davinci_003_results, width = 0.5, color = ["red", "green", "blue", "purple"])
   
   ax[0, 0].set_title('Human Participants')
   ax[0, 0].set_ylabel('Passing Rate')
   ax[0, 1].set_title('gpt-4')
   ax[0, 1].set_ylabel('Passing Rate')
   ax[1, 0].set_title('gpt-3.5-turbo')
   ax[1, 0].set_ylabel('Passing Rate')
   ax[1, 1].set_title('text-davinci-003')
   ax[1, 1].set_ylabel('Passing Rate')
   
   for ax in ax.flat:
       ax.set_ylim(0, 105)
        
   plt.subplots_adjust(hspace = 0.4, wspace = 0.3)
   
   plt.show()
   #plt.savefig('PpQ_subplots.png', dpi = 300)
   
def stat_tests():
    # Per ToM story
    print("---------------------------------------------")
    print("Per ToM story")
    UTT, UCT, DBT = preprocess_ToM_stories()

    means_story = [np.mean(UTT[0, :]), np.mean(UCT[0, :]), np.mean(DBT[0, :])]
    text_davinci_003_results = [UTT[1, :], UCT[1, :], DBT[1, :]]
    gpt_3_5_results = [UTT[2, :], UCT[2, :], DBT[2, :]]
    gpt_4_results = [UTT[3, :], UCT[3, :], DBT[3, :]]
    
    # text-davinci-003
    _, p_value_1_1 = stats.ttest_1samp(a = text_davinci_003_results[0], popmean = means_story[0])
    _, p_value_1_2 = stats.ttest_1samp(a = text_davinci_003_results[1], popmean = means_story[1])
    _, p_value_1_3 = stats.ttest_1samp(a = text_davinci_003_results[2], popmean = means_story[2])
    print("text-davinci-003: ", round(p_value_1_1, 3), round(p_value_1_2, 3), round(p_value_1_3, 3))
    
    # gpt-3.5-turbo
    _, p_value_2_1 = stats.ttest_1samp(a = gpt_3_5_results[0], popmean = means_story[0])
    _, p_value_2_2 = stats.ttest_1samp(a = gpt_3_5_results[1], popmean = means_story[1])
    _, p_value_2_3 = stats.ttest_1samp(a = gpt_3_5_results[2], popmean = means_story[2])
    print("gpt-3.5-turbo: ", round(p_value_2_1, 3), round(p_value_2_2, 3), round(p_value_2_3, 3))
    
    # gpt-4
    _, p_value_3_1 = stats.ttest_1samp(a = gpt_4_results[0], popmean = means_story[0])
    _, p_value_3_2 = stats.ttest_1samp(a = gpt_4_results[1], popmean = means_story[1])
    _, p_value_3_3 = stats.ttest_1samp(a = gpt_4_results[2], popmean = means_story[2])
    print("gpt-4: ", round(p_value_3_1, 3), round(p_value_3_2, 3), round(p_value_3_3, 3))
    print("---------------------------------------------")
    
    # Per ToM Question
    print("Per ToM question")
    reality, first_order, second_order, third_order = preprocess_ToM_questions()
    
    means_question = [np.mean(reality[0, :]), np.mean(first_order[0, :]), np.mean(second_order[0, :]), np.mean(third_order[0, :])]
    text_davinci_003_results = [reality[1, :], first_order[1, :], second_order[1, :],third_order[1, :]]
    gpt_3_5_results = [reality[2, :], first_order[2, :], second_order[2, :], third_order[2, :]]
    gpt_4_results = [reality[3, :], first_order[3, :], second_order[3, :], third_order[3, :]]

    # text-davinci-003
    _, p_value_1_1 = stats.ttest_1samp(a = text_davinci_003_results[0], popmean = means_question[0])
    _, p_value_1_2 = stats.ttest_1samp(a = text_davinci_003_results[1], popmean = means_question[1])
    _, p_value_1_3 = stats.ttest_1samp(a = text_davinci_003_results[2], popmean = means_question[2])
    _, p_value_1_4 = stats.ttest_1samp(a = text_davinci_003_results[3], popmean = means_question[3])
    print("text-davinci-003: ", round(p_value_1_1, 3), round(p_value_1_2, 3), round(p_value_1_3, 3), round(p_value_1_4, 3))
    
    # gpt-3.5-turbo
    _, p_value_2_1 = stats.ttest_1samp(a = gpt_3_5_results[0], popmean = means_question[0])
    _, p_value_2_2 = stats.ttest_1samp(a = gpt_3_5_results[1], popmean = means_question[1])
    _, p_value_2_3 = stats.ttest_1samp(a = gpt_3_5_results[2], popmean = means_question[2])
    _, p_value_2_4 = stats.ttest_1samp(a = gpt_3_5_results[3], popmean = means_question[3])
    print("gpt-3.5-turbo: ", round(p_value_2_1, 3), round(p_value_2_2, 3), round(p_value_2_3, 3), round(p_value_2_4, 3))
    
    # gpt-4
    warnings.filterwarnings("ignore", category = RuntimeWarning)
    _, p_value_3_1 = stats.ttest_1samp(a = gpt_4_results[0], popmean = means_question[0])
    _, p_value_3_2 = stats.ttest_1samp(a = gpt_4_results[1], popmean = means_question[1])
    _, p_value_3_3 = stats.ttest_1samp(a = gpt_4_results[2], popmean = means_question[2])
    _, p_value_3_4 = stats.ttest_1samp(a = gpt_4_results[3], popmean = means_question[3])
    print("gpt-4: ", round(p_value_3_1, 3), round(p_value_3_2, 3), round(p_value_3_3, 3), round(p_value_3_4, 3))
    print("---------------------------------------------")
    warnings.resetwarnings()
    