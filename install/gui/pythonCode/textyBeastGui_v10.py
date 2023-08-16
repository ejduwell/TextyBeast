#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext

# Define Functions
def select_input_dir():
    dir_selected = filedialog.askdirectory(title="Select Input Directory")
    input_dir_var.set(dir_selected)

def select_output_dir():
    dir_selected = filedialog.askdirectory(title="Select Output Directory")
    output_dir_var.set(dir_selected)

def select_param_file():
    file_selected = filedialog.askopenfilename(title="Select Parameter File")
    param_file_var.set(file_selected)

def run_command():
    global console_visible

    if not console_visible:
        console.pack(fill=tk.BOTH, expand=False)
        console_visible = True

    program = program_var.get()
    jobtype = jobtype_var.get()

    command = f"{program} {input_dir_var.get()} {output_dir_var.get()} {jobtype} {param_file_var.get()}"
    console.insert(tk.END, f"$ {command}\n")
    # Here you can run the actual command and capture its output, then display it in the console.
    # For now, it just shows the constructed command.

def update_canvas_text(*args):
    canvas.itemconfig(input_dir_label, text=input_dir_var.get())
    canvas.itemconfig(output_dir_label, text=output_dir_var.get())
    canvas.itemconfig(param_file_label, text=param_file_var.get())

def on_canvas_resized(event):
    canvas_width = event.width
    canvas_height = event.height
    sections = 7

    # Update dropdown and text positions
    canvas.coords(dropdown_window, canvas_width / 2, canvas_height * 1 / sections)
    canvas.coords(dropdown_label, canvas_width / 2, canvas_height * 1 / sections - 35)
    canvas.coords(dropdown_jobtype_window, canvas_width / 2, canvas_height * 2 / sections)
    canvas.coords(jobtype_dropdown_label, canvas_width / 2, canvas_height * 2 / sections - 35)
    canvas.coords(select_input_btn, canvas_width / 2, canvas_height * 3 / sections)
    canvas.coords(select_output_btn, canvas_width / 2, canvas_height * 4 / sections)
    canvas.coords(select_param_btn, canvas_width / 2, canvas_height * 5 / sections)
    canvas.coords(run_btn, canvas_width / 2, canvas_height * 6 / sections)
    
    # Update text labels positions for directories and files
    canvas.coords(input_dir_label, canvas_width / 2, canvas_height * 3 / sections + 20) # Centered below the "Select Input Directory" button
    canvas.coords(output_dir_label, canvas_width / 2, canvas_height * 4 / sections + 20) # Centered below the "Select Output Directory" button
    canvas.coords(param_file_label, canvas_width / 2, canvas_height * 5 / sections + 20) # Centered below the "Select Parameter File" button

root = tk.Tk()
root.title("TextyBeast Job Runner")
#root.geometry("600x420")
root.geometry("800x1000")

root.configure(bg="#d9d9d9")

canvas = tk.Canvas(root, bg="#d9d9d9")
canvas.pack(fill=tk.BOTH, expand=True)
canvas.bind("<Configure>", on_canvas_resized)

style = ttk.Style()
style.configure("TOptionMenu", background='#ccc')

dropdown_label = canvas.create_text(300, 40, text="Choose a Function:", anchor=tk.N, fill="black", font=("Arial", 10, "bold"))

program_var = tk.StringVar(value="Select...")
programs = ["Select...", "textyBeast_localjob", "textyBeast_slurmjob", "textyBeast_remotejob"]
dropdown = ttk.OptionMenu(root, program_var, *programs)
dropdown_window = canvas.create_window(300, 85, window=dropdown, anchor=tk.CENTER)

jobtype_dropdown_label = canvas.create_text(300, 170, text="Choose a Job Type:", anchor=tk.N, fill="black", font=("Arial", 10, "bold"))
jobtype_var = tk.StringVar(value="Select...")
jobtypes = ["Select...", "jobtype1", "jobtype2", "jobtype3"]
dropdown_jobtype = ttk.OptionMenu(root, jobtype_var, *jobtypes)
dropdown_jobtype_window = canvas.create_window(300, 215, window=dropdown_jobtype, anchor=tk.CENTER)

# Text variables for directories and files
input_dir_var = tk.StringVar()
output_dir_var = tk.StringVar()
param_file_var = tk.StringVar()

# Bind our update function to text variable changes
input_dir_var.trace_add("write", update_canvas_text)
output_dir_var.trace_add("write", update_canvas_text)
param_file_var.trace_add("write", update_canvas_text)

# Text labels for directories and files
input_dir_label = canvas.create_text(300, 285, text="", anchor=tk.CENTER)
output_dir_label = canvas.create_text(300, 345, text="", anchor=tk.CENTER)
param_file_label = canvas.create_text(300, 405, text="", anchor=tk.CENTER)

# Buttons
select_input_btn = canvas.create_window(300, 260, window=ttk.Button(canvas, text="Select Input Directory", command=select_input_dir))
select_output_btn = canvas.create_window(300, 320, window=ttk.Button(canvas, text="Select Output Directory", command=select_output_dir))
select_param_btn = canvas.create_window(300, 380, window=ttk.Button(canvas, text="Select Parameter File", command=select_param_file))
run_btn = canvas.create_window(300, 440, window=ttk.Button(canvas, text="Run", command=run_command))

# Create the console (hidden by default)
console_visible = False
console = scrolledtext.ScrolledText(root, bg='black', fg='green')
console.pack_forget()

root.mainloop()
