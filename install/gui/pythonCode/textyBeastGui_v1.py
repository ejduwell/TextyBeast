#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog

def submit_command():
    selected_function = function_dropdown.get()
    input_dir = input_dir_entry.get()
    output_dir = output_dir_entry.get()
    jobtype_selected = jobtype_dropdown.get()
    param_file = filedialog.askopenfilename(title="Select Parameter File")

    command = f"{selected_function} {input_dir} {output_dir} \"{jobtype_selected}\" \"{param_file}\""
    
    # TODO: Run the command using subprocess or os.system or a method of your choice
    print(command)  # For now, just print the command

root = tk.Tk()
root.title("TextyBeast Job Runner")

# Dropdown for command selection
FUNCTIONS = ["textyBeast_localjob", "textyBeast_slurmjob", "textyBeast_remotejob"]
function_dropdown_label = tk.Label(root, text="Choose a function:")
function_dropdown_label.pack()
function_dropdown = tk.StringVar(root)
function_dropdown.set(FUNCTIONS[0])  # default value
function_dd = tk.OptionMenu(root, function_dropdown, *FUNCTIONS)
function_dd.pack()

# Input directory selection
input_dir_label = tk.Label(root, text="Input Directory:")
input_dir_label.pack()
input_dir_entry = tk.Entry(root, width=50)
input_dir_entry.pack()

# Output directory selection
output_dir_label = tk.Label(root, text="Output Directory:")
output_dir_label.pack()
output_dir_entry = tk.Entry(root, width=50)
output_dir_entry.pack()

# Jobtype dropdown selection
JOBTYPES = ["vl", "di"]  # Replace with your actual job types
jobtype_dropdown_label = tk.Label(root, text="Choose a job type:")
jobtype_dropdown_label.pack()
jobtype_dropdown = tk.StringVar(root)
jobtype_dropdown.set(JOBTYPES[0])  # default value
jobtype_dd = tk.OptionMenu(root, jobtype_dropdown, *JOBTYPES)
jobtype_dd.pack()

# Button to run the command
submit_button = tk.Button(root, text="Run", command=submit_command)
submit_button.pack()

root.mainloop()
