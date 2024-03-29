# Maksim Terentev
# GPT UI
# Last changes: 21/06/2023
# Version 1.3.5

import openai
import re
import requests
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from pandastable import Table
from GPT_API import *
from Auxiliary import *


# PLEASE, SET THE DEFAULT KEY HERE
default_key = "YOUR_KEY_HERE"


class GPT_UI:
    def __init__(self):
        self.df_ToM_tests = pd.DataFrame() # Original tests file
        self.df_ToM_tests_results = pd.DataFrame() # Modified tests file
        
        # Set up the frame
        self.root = tk.Tk()
        self.root.title("GPT UI")
        self.root['background'] = "#bcd4cc"
        frame = tk.Frame(self.root, bg = "#bcd4cc")
        frame.pack()
        
        ##### Personal Key field #####
        key_frame = tk.LabelFrame(frame, text = "1. Personal Key", font = ("Arial", 16, "bold"), bg = "#bcd4cc") # Specify frame
        key_frame.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = tk.W) # Specify grid
        
        provide_key_label = tk.Label(key_frame, text = "Enter OpenAI personal key:", bg = "#bcd4cc") # Specify label
        provide_key_label.grid(row = 0, column = 0, padx = 5, pady = 0, sticky = tk.W)
        self.key_entry = tk.Entry(key_frame, show = '*', width = 35, highlightbackground = "#bcd4cc") # Specify entry
        self.key_entry.grid(row = 1, column = 0, padx = 5, pady = 0, sticky = tk.W)

        self.key_option_var = tk.BooleanVar() # Variable for determining the state of the check button
        self.def_key_checkbtn = tk.Checkbutton(key_frame, text = "Use defualt key", variable = self.key_option_var, 
                                               bg = "#bcd4cc", command = self.set_default_key) # Specify check button
        self.def_key_checkbtn.grid(row = 2, column = 0, padx = 5, pady = 5, sticky = tk.W)
        self.submit_key_btn = tk.Button(key_frame, text = "OK", highlightbackground = "#bcd4cc", command = self.read_key) # Specify button
        self.submit_key_btn.grid(row = 1, column = 1, padx = 0, pady = 0)
        self.change_key_btn = tk.Button(key_frame, text = "Change", highlightbackground = "#bcd4cc", command = self.change_key)
        self.change_key_btn.grid(row = 1, column = 2, padx = 5, pady = 0)
        
        # Default states
        self.key_option_var.set(False)
        self.key_entry.config(state = "normal")
        self.submit_key_btn.config(state = "normal")
        self.change_key_btn.config(state = "disabled")
        ########################################################################

        ##### Read ToM Test(s) field #####
        read_tests_frame = tk.LabelFrame(frame, text = "2. Read ToM Test(s)", font = ("Arial", 16, "bold"), bg = "#bcd4cc")
        read_tests_frame.grid(row = 1, column = 0, padx = 10, pady = 0, sticky = tk.W) 
        
        # Two inner frames
        read_CSV_frame = tk.LabelFrame(read_tests_frame, text = "", font = ("Arial", 15), bg = "#bcd4cc")
        read_CSV_frame.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = tk.N) 
        insert_test_manually_frame = tk.LabelFrame(read_tests_frame, text = "", font = ("Arial", 15), bg = "#bcd4cc")
        insert_test_manually_frame.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = tk.N) 
        
        # Choice radiobuttons
        self.read_choice_var = tk.IntVar() # Variable for determining the state of the radio button
        self.read_CSV_rbtn = tk.Radiobutton(read_CSV_frame, text = 'Read CSV file', variable = self.read_choice_var, 
                                            value = 1, command = self.read_or_insert_tests, background = "#bcd4cc") # Specify radio button
        self.read_CSV_rbtn.grid(row = 0, column = 0, padx = 5, pady = 5)
        self.insert_test_manually_rbtn = tk.Radiobutton(insert_test_manually_frame, text = 'Insert test manually', value = 2, background = "#bcd4cc",
                                               variable = self.read_choice_var, command = self.read_or_insert_tests)
        self.insert_test_manually_rbtn.grid(row = 0, column = 1, padx = 5, pady = 5, sticky = tk.W)
        
        # Read CSV frame
        self.file_name_label = tk.Label(read_CSV_frame, text = "", font = ("Helvetica", "13", "bold"), bg = "#bcd4cc", width = 20)
        self.file_name_label.grid(row = 1, column = 1, padx = 5, pady = 0, sticky = tk.W + tk.N)
        self.browse_file_btn = tk.Button(read_CSV_frame, text = "Browse file", highlightbackground = "#bcd4cc", command = self.read_CSV)
        self.browse_file_btn.grid(row = 2, column = 0, padx = 5, pady = 5, sticky = tk.W + tk.N)
        self.show_tests_CSV_btn = tk.Button(read_CSV_frame, text = "Show tests", highlightbackground = "#bcd4cc", command = self.show_tests)
        self.show_tests_CSV_btn.grid(row = 2, column = 1, padx = 5, pady = 5, sticky = tk.N)
        
        # Insert test manually frame
        id_label = tk.Label(insert_test_manually_frame, text = "ID:", bg = "#bcd4cc")
        id_label.grid(row = 1, column = 0, padx = 0, pady = 0, sticky = tk.E)
        self.id_text = tk.Text(insert_test_manually_frame, highlightbackground = "#bcd4cc", height = 1, width = 10)
        self.id_text.grid(row = 1, column = 1, padx = 5, pady = 0, sticky = tk.W)
        
        description_label = tk.Label(insert_test_manually_frame, text = "Description:", bg = "#bcd4cc")
        description_label.grid(row = 2, column = 0, padx = 0, pady = 0, sticky = tk.E)
        self.description_test_text = tk.Text(insert_test_manually_frame, highlightbackground = "#bcd4cc", height = 6, width = 40)
        self.description_test_text.grid(row = 2, column = 1, padx = 5, pady = 0, sticky = tk.W, rowspan = 5)
        
        question_label = tk.Label(insert_test_manually_frame, text = "Question:", bg = "#bcd4cc")
        question_label.grid(row = 7, column = 0, padx = 0, pady = 0, sticky = tk.E)
        self.question_text = tk.Text(insert_test_manually_frame, highlightbackground = "#bcd4cc", height = 2, width = 40)
        self.question_text.grid(row = 7, column = 1, padx = 5, pady = 0, sticky = tk.W, rowspan = 2)
        
        correct_answer_label = tk.Label(insert_test_manually_frame, text = "Correct Answer:", bg = "#bcd4cc")
        correct_answer_label.grid(row = 9, column = 0, padx = 0, pady = 0, sticky = tk.E)
        self.correct_answer_text = tk.Text(insert_test_manually_frame, highlightbackground = "#bcd4cc", height = 2, width = 40)
        self.correct_answer_text.grid(row = 9, column = 1, padx = 5, pady = 0, sticky = tk.W)
        
        self.add_test_btn = tk.Button(insert_test_manually_frame, text = "Add test", highlightbackground = "#bcd4cc", command = self.add_test)
        self.add_test_btn.grid(row = 10, column = 1, padx = 5, pady = 5, sticky = tk.W)
        self.show_test_manual_btn = tk.Button(insert_test_manually_frame, text = "Show test", highlightbackground = "#bcd4cc", command = self.show_tests)
        self.show_test_manual_btn.grid(row = 10, column = 1, padx = 5, pady = 5)
        
        # Default states
        self.read_choice_var.set(1)
        self.read_CSV_rbtn.config(state = "disable")
        self.show_test_manual_btn.config(state = "disabled")
        self.insert_test_manually_rbtn.config(state = "disable")
        self.browse_file_btn.config(state = "disable")
        self.id_text.config(state = "disable")
        self.description_test_text.config(state = "disabled")
        self.question_text.config(state = "disabled")
        self.add_test_btn.config(state = "disabled")
        self.correct_answer_text.config(state = "disabled")
        self.show_tests_CSV_btn.config(state = "disabled")
        self.show_test_manual_btn.config(state = "disabled")
        ########################################################################
        
        #### Run ToM Test(s) ####
        run_tests_frame = tk.LabelFrame(frame, text = "3. Run ToM Test(s)", font = ("Arial", 16, "bold"), bg = "#bcd4cc")
        run_tests_frame.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = tk.W) 
        
        instruction_label = tk.Label(run_tests_frame, text = "Instruction", bg = "#bcd4cc")
        instruction_label.grid(row = 0, column = 0, padx = 5, pady = 0, sticky = tk.W)
        self.instruction_description_text = tk.Text(run_tests_frame, highlightbackground = "#bcd4cc", height = 3, width = 98)
        self.instruction_description_text.insert(tk.INSERT, "You will be given a story and provided with a question. Please, answer the question as accurately as possible. For yes/no questions, respond only with a 'yes' or a 'no'. For open questions, use a maximum of 10 words.")
        self.instruction_description_text.grid(row = 1, column = 0, padx = 5, pady = 0, columnspan = 4, sticky = tk.W)
        
        self.number_of_tests_var = tk.IntVar() # Variable for determining the number of tests
        number_of_tests_label = tk.Label(run_tests_frame, text = "No. questions", bg = "#bcd4cc")
        number_of_tests_label.grid(row = 2, column = 0, padx = 5, pady = 0)
        self.number_of_tests_entry = tk.Entry(run_tests_frame, highlightbackground = "#bcd4cc", width = 2, textvariable = self.number_of_tests_var)
        self.number_of_tests_entry.grid(row = 3, column = 0, padx = 5, pady = 0)
       
        model_label = tk.Label(run_tests_frame, text = "Choose the model", bg = "#bcd4cc")
        model_label.grid(row = 2, column = 1, padx = 5, pady = 0)
        options_list = ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]
        self.model_var = tk.StringVar() # Variable for determining the model option
        self.model_menu = tk.OptionMenu(run_tests_frame, self.model_var, *options_list) # Specify open menu
        self.model_menu.config(bg = "#bcd4cc", width = 10)
        self.model_menu.grid(row = 3, column = 1, padx = 5, pady = 0)

        temperature_label = tk.Label(run_tests_frame, text = "Set the temperature", bg = "#bcd4cc")
        temperature_label.grid(row = 2, column = 2, padx = 0, pady = 0)
        self.temperature_var = tk.IntVar()  # Variable for determining the temperature parameter
        self.temperature_spinbox = tk.Spinbox(run_tests_frame, width = 3, from_ = 0, to = 1, increment = .1, 
                                              highlightbackground = "#bcd4cc", textvariable = self.temperature_var) # Specify spin box
        self.temperature_spinbox.grid(row = 3, column = 2, padx = 0, pady = 0)
        
        set_max_tokens_label = tk.Label(run_tests_frame, text = "Set the max number of tokens", bg = "#bcd4cc")
        set_max_tokens_label.grid(row = 2, column = 3, padx = 0, pady = 0)
        self.max_tokens_var = tk.IntVar() # Variable for determining the max tokens parameter
        self.max_tokens_entry = tk.Entry(run_tests_frame, width = 3, highlightbackground = "#bcd4cc", textvariable = self.max_tokens_var)
        self.max_tokens_entry.grid(row = 3, column = 3, padx = 0, pady = 0)
        
        self.run_experiment_btn = tk.Button(run_tests_frame, text = "Run experiment", highlightbackground = "#bcd4cc", command = self.run_experiment)
        self.run_experiment_btn.grid(row = 4, column = 2, padx = 0, pady = 0)
        
        self.show_results_btn = tk.Button(run_tests_frame, text = "Show results", highlightbackground = "#bcd4cc", command = self.show_results)
        self.show_results_btn.grid(row = 5, column = 2, padx = 0, pady = 5)
        
        # Default states
        self.number_of_tests_var.set(3)
        self.model_var.set("gpt-3.5-turbo")
        self.temperature_var.set(0)
        self.max_tokens_var.set(30)
        self.instruction_description_text.config(state = "disabled")
        self.number_of_tests_entry.config(state = "disabled")
        self.model_menu.config(state = "disabled")
        self.temperature_spinbox.config(state = "disabled")
        self.max_tokens_entry.config(state = "disabled")
        self.run_experiment_btn.config(state = "disabled")
        self.show_results_btn.config(state = "disabled")
        ########################################################################
        
        ##### Save Results #####
        save_results_frame = tk.LabelFrame(frame, text = "4. Save Results", font = ("Arial", 16, "bold"), bg = "#bcd4cc")
        save_results_frame.grid(row = 3, column = 0, padx = 10, pady = 5, sticky = tk.W) 
        
        self.save_file_name_var = tk.StringVar() # Variable for determining the name of the file that is being saved
        self.save_file_name_entry = tk.Entry(save_results_frame, highlightbackground = "#bcd4cc", width = 20, textvariable = self.save_file_name_var)
        self.save_file_name_entry.grid(row = 0, column = 0, padx = 5, pady = 5, sticky = tk.E)
        self.save_results_btn = tk.Button(save_results_frame, text = "Save", highlightbackground = "#bcd4cc", command = self.save_CSV)
        self.save_results_btn.grid(row = 0, column = 1, padx = 5, pady = 5, sticky = tk.W)
        
        # Default states
        self.save_file_name_var.set("ToM_results.csv")
        self.save_results_btn.config(state = "disabled")
        self.save_file_name_entry.config(state = "disabled")
        ########################################################################
        
        ##### Other enteties #####
        self.close_btn = tk.Button(frame, text = "Close", highlightbackground = "#bcd4cc", fg = "red", command = self.close)
        self.close_btn.grid(row = 0, column = 0, padx = 50, pady = 20, sticky = tk.E + tk.N)
          
        self.reset_btn = tk.Button(frame, text = "Reset", highlightbackground = "#bcd4cc", bg = "orange", command = self.reset)
        self.reset_btn.grid(row = 0, column = 0, padx = 150, pady = 20, sticky = tk.E + tk.N)
        
        self.performance_plot_btn = tk.Button(frame, text = "Performance plot", highlightbackground = "#bcd4cc", command = self.performance_plot)
        self.performance_plot_btn.grid(row = 0, column = 0, padx = 65, pady = 25, sticky = tk.E + tk.S)
        
        developer_info_label = tk.Label(frame, text = "Developed by Maksim Terentev", bg = "#bcd4cc", fg = "#70998B")
        developer_info_label.grid(row = 4, column = 0, padx = 0, pady = 5)
        ########################################################################
        
        self.root.mainloop() # Main loop
    
    ## Methods ##
    # Sets the default key entered above, as the personal OpenAI key
    def set_default_key(self):
        # If the "Use default key" check button has been selected
        if self.key_option_var.get():
            # Check whether the format is correct and the key is valid
            if not is_valid_api_key_format(default_key):
                messagebox.showerror("Window", "Invalid OpenAI key format!")
                self.key_option_var.set(False)
            elif not check_api_key_authorization(default_key):
                messagebox.showerror("Window", "Unauthorized OpenAI key!") 
                self.key_option_var.set(False)  
            else:
                openai.api_key = default_key
                self.key_entry.config(state = "normal")
                self.key_entry.delete(0, "end")
                self.key_entry.config(state = "disabled")
                self.submit_key_btn.config(state = "disabled")
                self.change_key_btn.config(state = "disabled")
                # Enable all actions
                self.read_CSV_rbtn.config(state = "normal")
                self.insert_test_manually_rbtn.config(state = "normal")
                if self.read_choice_var.get() == 1:
                    self.browse_file_btn.config(state = "normal")  
                if self.read_choice_var.get() == 2:
                    self.insert_test_manually_rbtn.config(state = "normal")
                    self.id_text.config(state = "normal")
                    self.description_test_text.config(state = "normal")
                    self.question_text.config(state = "normal")
                    self.correct_answer_text.config(state = "normal")
                    self.add_test_btn.config(state = "normal")
                if self.df_ToM_tests.size != 0:
                    self.show_tests_CSV_btn.config(state = "normal")
                    self.show_test_manual_btn.config(state = "normal")
                    self.instruction_description_text.config(state = "normal")
                    self.number_of_tests_entry.config(state = "normal")
                    self.model_menu.config(state = "normal")
                    self.temperature_spinbox.config(state = "normal")
                    self.max_tokens_entry.config(state = "normal")
                    self.run_experiment_btn.config(state = "normal")
                    if self.df_ToM_tests_results["Answer 1"].any():
                        self.show_results_btn.config(state = "normal")
                        self.save_file_name_entry.config(state = "normal")
                        self.save_results_btn.config(state = "normal")
        # If the "Use default key" check button has been deselected
        else:
            self.key_entry.config(state = "normal")
            self.key_entry.delete(0, "end")
            self.submit_key_btn.config(state = "normal")
            # Disable all actions
            # Read/Insert ToM Test(s) field
            self.read_CSV_rbtn.config(state = "disabled")
            self.browse_file_btn.config(state = "disabled")
            self.insert_test_manually_rbtn.config(state = "disabled")
            self.id_text.config(state = "disable")
            self.description_test_text.config(state = "disabled")
            self.question_text.config(state = "disabled")
            self.correct_answer_text.config(state = "disabled")
            self.add_test_btn.config(state = "disabled")
            self.show_tests_CSV_btn.config(state = "disabled")
            self.show_test_manual_btn.config(state = "disabled")

            # Run ToM Test(s) field
            self.instruction_description_text.config(state = "disabled")
            self.number_of_tests_entry.config(state = "disabled")
            self.model_menu.config(state = "disabled")
            self.temperature_spinbox.config(state = "disabled")
            self.max_tokens_entry.config(state = "disabled")
            self.run_experiment_btn.config(state = "disabled")
            self.show_results_btn.config(state = "disabled")
            # Save Results field
            self.save_file_name_entry.config(state = "disabled")
            self.save_results_btn.config(state = "disabled")
            
    # Reads and sets the key 
    def read_key(self):
        key = self.key_entry.get()
        # Check whether the format is correct and the key is valid
        if not is_valid_api_key_format(key):
            messagebox.showerror("Window", "Invalid OpenAI key format!")
            self.key_entry.delete(0, "end")
        elif not check_api_key_authorization(key):
            messagebox.showerror("Window", "Unauthorized OpenAI key!")    
            self.key_entry.delete(0, "end")
        else:
            openai.api_key = key
            messagebox.showinfo("Window", "The key has been successfully set!")
            self.submit_key_btn.config(state = "disabled")
            self.key_entry.config(state = "disabled")
            self.change_key_btn.config(state = "normal")
            
            self.read_CSV_rbtn.config(state = "normal")
            self.insert_test_manually_rbtn.config(state = "normal")
            if self.read_choice_var.get() == 1:
                    self.browse_file_btn.config(state = "normal")   
    
    # Allows the key to be changed 
    def change_key(self):
        self.change_key_btn.config(state = "disabled")
        self.key_entry.config(state = "normal")
        self.key_entry.delete(0, "end")
        self.submit_key_btn.config(state = "normal")
    
    # Allows to choose between reading and inserting tests and sets necessary parameters 
    def read_or_insert_tests(self):
        self.instruction_description_text.config(state = "disabled")
        self.number_of_tests_entry.config(state = "disabled")
        self.model_menu.config(state = "disabled")
        self.temperature_spinbox.config(state = "disabled")
        self.max_tokens_entry.config(state = "disabled")
        self.show_tests_CSV_btn.config(state = "disabled")
        self.show_test_manual_btn.config(state = "disabled")
        self.run_experiment_btn.config(state = "disabled")
        self.show_results_btn.config(state = "disabled")
        self.save_file_name_entry.config(state = "disabled")
        self.save_results_btn.config(state = "disabled")
        # Read CSV
        if self.read_choice_var.get() == 1:
            self.df_ToM_tests = pd.DataFrame()
            self.df_ToM_tests_results = pd.DataFrame()
            
            self.id_text.delete("1.0", "end")
            self.description_test_text.delete("1.0", "end")
            self.question_text.delete("1.0", "end")
            self.correct_answer_text.delete("1.0", "end")
            
            self.browse_file_btn.config(state = "normal")
            self.id_text.config(state = "disable")
            self.description_test_text.config(state = "disabled")
            self.question_text.config(state = "disabled")
            self.correct_answer_text.config(state = "disabled")
            self.add_test_btn.config(state = "disabled")
            
            self.number_of_tests_var.set(3)

        # Insert test manually
        if self.read_choice_var.get() == 2:
            self.df_ToM_tests = pd.DataFrame()
            self.df_ToM_tests_results = pd.DataFrame()
            
            self.browse_file_btn.config(state = "disable")
            self.file_name_label.config(text = "")
            
            self.id_text.config(state = "normal")
            self.description_test_text.config(state = "normal")
            self.question_text.config(state = "normal")
            self.correct_answer_text.config(state = "normal")
            self.add_test_btn.config(state = "normal")
            
            self.number_of_tests_var.set(1)
        
    # Reads the CSV file into the pandas DataFrame
    def read_CSV(self):
        # Open the CSV file
        f_types = [('CSV files', "*.csv"),('All', "*.*")]
        file = tk.filedialog.askopenfilename(filetypes = f_types)
        if not file:
            messagebox.showerror("Window", "The file hasn't been selected!")
        else:
            self.df_ToM_tests = pd.read_csv(file, sep = ';')
            # Check whether it has the correct columns in the proper order
            if set(["ID", "Description", "Question 1", "Answer 1", "Correct Answer 1"]).issubset(self.df_ToM_tests.columns):
                messagebox.showinfo("Window", "The file has been sucsesfully read!")
                self.df_ToM_tests_results  = self.df_ToM_tests 
                filename = file.split('/')[len(file.split('/'))-1]
                self.file_name_label['text'] = filename
                
                self.show_tests_CSV_btn.config(state = "normal")
                self.show_test_manual_btn.config(state = "normal")
                self.instruction_description_text.config(state = "normal")
                self.number_of_tests_var.set(3)
                self.number_of_tests_entry.config(state = "normal")
                self.model_menu.config(state = "normal")
                self.temperature_spinbox.config(state = "normal")
                self.max_tokens_entry.config(state = "normal")
                self.run_experiment_btn.config(state = "normal")
                self.save_file_name_entry.config(state = "disabled")
                self.save_results_btn.config(state = "disabled")
                
            elif self.df_ToM_tests.empty == True:
                messagebox.showerror("Window", "The file is empty! Select another file or enter the test manually.")
            else:
                messagebox.showerror("Window", "The file has the wrong structure or column names.")
            
    # Adds test manually
    def add_test(self):
        if self.id_text.get("1.0", 'end-1c') != "" and self.description_test_text.get("1.0", 'end-1c') != "" and self.question_text.get("1.0", 'end-1c') != "" and self.correct_answer_text.get("1.0", 'end-1c') != "":
            self.df_ToM_tests = pd.DataFrame(columns = ['ID', 'Description', 'Question 1', 'Answer 1', 'Correct Answer 1'])
            self.df_ToM_tests.loc[len(self.df_ToM_tests.index)] = [self.id_text.get("1.0", 'end-1c'), 
                                                                self.description_test_text.get("1.0", 'end-1c'), 
                                                                self.question_text.get("1.0", 'end-1c'), "", 
                                                                self.correct_answer_text.get("1.0", 'end-1c')] 
            self.df_ToM_tests_results = self.df_ToM_tests
            messagebox.showinfo("Window", "The test has been successfully added")
        
            self.id_text.delete("1.0", "end")
            self.description_test_text.delete("1.0", "end")
            self.question_text.delete("1.0", "end")
            self.correct_answer_text.delete("1.0", "end")
            
            self.show_tests_CSV_btn.config(state = "normal")
            self.show_test_manual_btn.config(state = "normal")
            self.instruction_description_text.config(state = "normal")
            self.number_of_tests_var.set(1)
            self.number_of_tests_entry.config(state = "normal")
            self.model_menu.config(state = "normal")
            self.temperature_spinbox.config(state = "normal")
            self.max_tokens_entry.config(state = "normal")
            self.run_experiment_btn.config(state = "normal")
        else:
            messagebox.showerror("Window", "The test hasn't been added! One or more of the entries is empty.")
            
    # Shows the read/inserted test(s)    
    def show_tests(self):
        new_window = tk.Toplevel()
        new_window.title("ToM test(s)")
        new_frame = tk.Frame(new_window, bg = "#bcd4cc")
        new_frame.pack()
        df_drame = tk.Frame(new_frame)
        df_drame.pack(fill = tk.BOTH, expand = 1)
        pt = Table(df_drame, dataframe = self.df_ToM_tests, showtoolbar = True, showstatusbar = True)
        pt.show()
    
    # Runs the experiment and stores the results in df_ToM_tests 
    def run_experiment(self):
        # In case the CSV was read
        if self.read_choice_var.get() == 1:
            n_tests = self.df_ToM_tests.shape[0]
            n_questions = self.number_of_tests_var.get()
            for test in range(n_tests):
                for question in range(n_questions):
                    if ('Question ' + str(question + 1)) not in self.df_ToM_tests or pd.isnull(self.df_ToM_tests.loc[test, 'Question ' + str(question + 1)]):
                        continue
                    response = conversation(model = self.model_var.get(), system_content = self.instruction_description_text.get("1.0", 'end-1c'), 
                                            max_tokens = self.max_tokens_var.get(), temperature = self.temperature_var.get(), 
                                            user_content = self.df_ToM_tests.loc[test, 'Description'] + '\n' + self.df_ToM_tests.loc[test, 'Question ' + str(question + 1)])
                    self.df_ToM_tests_results.loc[test, 'Answer ' + str(question + 1)] = response
        # In case the test was manually entered
        if self.read_choice_var.get() == 2:
            response = conversation(model = self.model_var.get(), system_content = self.instruction_description_text.get("1.0", 'end-1c'), 
                                            max_tokens = self.max_tokens_var.get(), temperature = self.temperature_var.get(), 
                                            user_content = self.df_ToM_tests.loc[0, 'Description'] + '\n' + self.df_ToM_tests.loc[0, 'Question 1'])
            self.df_ToM_tests_results.loc[0, 'Answer 1'] = response
       
        self.show_results_btn.config(state = "normal")
        self.save_file_name_entry.config(state = "normal")
        self.save_results_btn.config(state = "normal")
        
        messagebox.showinfo("Window", "The compilation has been sucsesfully finished!")

    # Shows the pandas DataFrame with results
    def show_results(self):
        new_window = tk.Toplevel()
        new_window.title("Result(s) of ToM test(s)")
        new_frame = tk.Frame(new_window, bg = "#bcd4cc")
        new_frame.pack()
        df_drame = tk.Frame(new_frame)
        df_drame.pack(fill = tk.BOTH, expand = 1)
        pt = Table(df_drame, dataframe = self.df_ToM_tests_results, showtoolbar = True, showstatusbar = True)
        pt.show()
    
    # Saves the results in the CSV file   
    def save_CSV(self):
        ### Add new location
        self.df_ToM_tests.to_csv(self.save_file_name_var.get(), index = False)
        messagebox.showinfo("Window", "The results have been sucsesfully saved!")
    
    # Shows the performance plot in the new window
    # Please, choose one of the plots
    def performance_plot(self): 
        # The performance plot based on the type of the ToM story
        # performance_per_story_type()
        # The performance plot based on the type of the ToM question
        performance_per_question_type()
        
    # Resets GUI to the default state
    def reset(self):
        # Clean the key and DataFrames
        openai.api_key = ""
        self.df_ToM_tests = pd.DataFrame()
        self.df_ToM_tests_results = pd.DataFrame()
       
        # Personal key field
        self.key_option_var.set(False)
        self.key_entry.config(state = "normal")
        self.key_entry.delete(0, 'end')
        self.key_entry.config(state = "normal")
        self.submit_key_btn.config(state = "normal")
        self.change_key_btn.config(state = "disabled")
        
        # Read/Insert ToM Test(s) field
        self.file_name_label.config(text = "")
        self.read_choice_var.set(1)
        self.read_CSV_rbtn.config(state = "disabled")
        self.browse_file_btn.config(state = "disabled")
        self.insert_test_manually_rbtn.config(state = "disabled")
        self.id_text.delete("1.0", "end")
        self.description_test_text.delete("1.0", "end")
        self.question_text.delete("1.0", "end")
        self.correct_answer_text.delete("1.0", "end")
        self.id_text.config(state = "disable")
        self.description_test_text.config(state = "disabled")
        self.question_text.config(state = "disabled")
        self.add_test_btn.config(state = "disabled")
        self.correct_answer_text.config(state = "disabled")
        self.show_tests_CSV_btn.config(state = "disabled")
        self.show_test_manual_btn.config(state = "disabled")
        
        # Run ToM Test(s) field
        self.instruction_description_text.delete('1.0', tk.END)
        self.instruction_description_text.insert(tk.INSERT, "You will be given a story and provided with a question. Please, answer the question as accurately as possible. For yes/no questions, respond only with a 'yes' or a 'no'. For open questions, use a maximum of 10 words.")
        self.number_of_tests_var.set(3)
        self.model_var.set("gpt-3.5-turbo")
        self.temperature_var.set(0.0)
        self.max_tokens_var.set(30)
        self.instruction_description_text.config(state = "disabled")
        self.number_of_tests_entry.config(state = "disabled")
        self.model_menu.config(state = "disabled")
        self.temperature_spinbox.config(state = "disabled")
        self.max_tokens_entry.config(state = "disabled")
        self.run_experiment_btn.config(state = "disabled")
        self.show_results_btn.config(state = "disabled")
        
        # Save Results field
        self.save_file_name_var.set("ToM_results.CSV")
        self.save_file_name_entry.config(state = "disabled")
        self.save_results_btn.config(state = "disabled")
    
    # Closes the application
    def close(self):
        self.root.quit()
   
##### Checking the format and validity of the entered key #####
def is_valid_api_key_format(key):
    return bool(re.match(r"^sk-[a-zA-Z0-9]{32,}$", key))

def check_api_key_authorization(key):
    headers = {"Authorization": f"Bearer {key}"}
    response = requests.get("https://api.openai.com/v1/engines", headers = headers)
    return response.status_code == 200
###############################################################




if __name__ == "__main__":
    GPT_UI()
        
     
   




    