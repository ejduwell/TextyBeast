#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import subprocess
import threading
import os
import socket
import sys

# Get input args
baseDir = sys.argv[1];

# Define Functions
def get_prompt():
    username = os.getlogin()
    computername = socket.gethostname()
    current_directory = os.getcwd()
    return f"{username}@{computername}:{current_directory}$ "

def select_input_dir():
    dir_selected = filedialog.askdirectory(title="Select Input Directory")
    input_dir_var.set(dir_selected)

def select_output_dir():
    dir_selected = filedialog.askdirectory(title="Select Output Directory")
    output_dir_var.set(dir_selected)

def select_param_file():
    pre_specified_directory = baseDir
    file_selected = filedialog.askopenfilename(title="Select Parameter File", initialdir=pre_specified_directory)
    if file_selected.startswith(pre_specified_directory):  # Ensure the file is from the pre-specified directory
        param_file_var.set(os.path.basename(file_selected))
    else:
        print("Please select a file from the specified directory.")

def get_system_path():
    try:
        result = subprocess.Popen("bash -l -c 'echo $PATH'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = result.communicate()
        return stdout.strip()
    except Exception:
        return None

def read_process_output(process, console):
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            hide_console_input()
            prompt = get_prompt()
            console.insert(tk.END, prompt)
            break
        if output:
            console.insert(tk.END, output)
            console.see(tk.END)  # Scroll to the end
    process.poll()

def send_input_to_process(process, input_data):
    process.stdin.write(input_data + '\n')
    process.stdin.flush()

def show_console_input():
    input_field.pack(fill=tk.X, padx=10, pady=(5, 10), expand=True)

def run_command():
    global console_visible

    if not console_visible:
        console.pack(fill=tk.BOTH, expand=True)
        console_visible = True

    show_console_input()  # Always show the input field when running a command

    program = program_var.get()
    jobtype = jobtype_var.get()
    command = f"{program} {input_dir_var.get()} {output_dir_var.get()} {jobtype} {param_file_var.get()}"
    console.insert(tk.END, f"$ {command}\n")
    system_path = get_system_path()
    try:
        bash_command = f"bash -c \"{command}\""
        env = {"PATH": system_path} if system_path else None
        
        process = subprocess.Popen(
            bash_command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        thread = threading.Thread(target=read_process_output, args=(process, console))
        thread.daemon = True
        thread.start()

        def on_enter_pressed(event):
            user_input = input_field.get()
            console.insert(tk.END, user_input + '\n')
            send_input_to_process(process, user_input)
            input_field.delete(0, tk.END)

        input_field.bind('<Return>', on_enter_pressed)
    except Exception as e:
        console.insert(tk.END, f"Failed to run command: {e}\n")

def update_canvas_text(*args):
    canvas.itemconfig(input_dir_label, text=input_dir_var.get())
    canvas.itemconfig(output_dir_label, text=output_dir_var.get())
    canvas.itemconfig(param_file_label, text=param_file_var.get())

def on_canvas_resized(event):
    canvas_width = event.width
    canvas_height = event.height
    sections = 7
    canvas.coords(dropdown_window, canvas_width / 2, canvas_height * 1 / sections)
    canvas.coords(dropdown_label, canvas_width / 2, canvas_height * 1 / sections - 35)
    canvas.coords(dropdown_jobtype_window, canvas_width / 2, canvas_height * 2 / sections)
    canvas.coords(jobtype_dropdown_label, canvas_width / 2, canvas_height * 2 / sections - 35)
    canvas.coords(select_input_btn, canvas_width / 2, canvas_height * 3 / sections)
    canvas.coords(select_output_btn, canvas_width / 2, canvas_height * 4 / sections)
    canvas.coords(select_param_btn, canvas_width / 2, canvas_height * 5 / sections)
    canvas.coords(run_btn, canvas_width / 2, canvas_height * 6 / sections)
    canvas.coords(input_dir_label, canvas_width / 2, canvas_height * 3 / sections + 20)
    canvas.coords(output_dir_label, canvas_width / 2, canvas_height * 4 / sections + 20)
    canvas.coords(param_file_label, canvas_width / 2, canvas_height * 5 / sections + 20)

def hide_console_input():
    input_field.pack_forget()

root = tk.Tk()
root.title("TextyBeast Job Runner")
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
jobtypes = ["Select...", "vl", "di"]
dropdown_jobtype = ttk.OptionMenu(root, jobtype_var, *jobtypes)
dropdown_jobtype_window = canvas.create_window(300, 215, window=dropdown_jobtype, anchor=tk.CENTER)
input_dir_var = tk.StringVar()
output_dir_var = tk.StringVar()
param_file_var = tk.StringVar()
input_dir_var.trace_add("write", update_canvas_text)
output_dir_var.trace_add("write", update_canvas_text)
param_file_var.trace_add("write", update_canvas_text)
input_dir_label = canvas.create_text(300, 285, text="", anchor=tk.CENTER)
output_dir_label = canvas.create_text(300, 345, text="", anchor=tk.CENTER)
param_file_label = canvas.create_text(300, 405, text="", anchor=tk.CENTER)
select_input_btn = canvas.create_window(300, 260, window=ttk.Button(canvas, text="Select Input Directory", command=select_input_dir))
select_output_btn = canvas.create_window(300, 320, window=ttk.Button(canvas, text="Select Output Directory", command=select_output_dir))
select_param_btn = canvas.create_window(300, 380, window=ttk.Button(canvas, text="Select Parameter File", command=select_param_file))
run_btn = canvas.create_window(300, 440, window=ttk.Button(canvas, text="Run", command=run_command))
console_visible = False
console = scrolledtext.ScrolledText(root, bg='black', fg='green')
console.pack_forget()
input_field = ttk.Entry(root)
input_field.pack_forget()
root.mainloop()
