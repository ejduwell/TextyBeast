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

# Set Global Variables:
# ----------------------------------------------------------------------------------------
callCountr=0 # initialize for counting calls in dbMode..
drifting_active = False # global variable to control the drifting animation loop
dbMode=1 # if = 1: turnos on debug mode to display call db printouts on command line...
# ----------------------------------------------------------------------------------------


# Define Functions
# ----------------------------------------------------------------------------------------
def create_terminal(canvas):
    terminal = tk.Text(canvas, bg="darkblue", fg="green", font=("Courier", 10), width=75, height=22)
    
    text_block = [
        " #########################################################################",
        " #########################################################################",
        " ##                                                                     ##",
        " ##        ::::::::::::::::::::::::    :::::::::::::::::   :::          ##", 
        " ##           :+:    :+:       :+:    :+:    :+:    :+:   :+:           ##",
        " ##          +:+    +:+        +:+  +:+     +:+     +:+ +:+             ##",
        " ##         +#+    +#++:++#    +#++:+      +#+      +#++:               ##",
        " ##        +#+    +#+        +#+  +#+     +#+       +#+                 ##",
        " ##       #+#    #+#       #+#    #+#    #+#       #+#                  ##",
        " ##      ###    #############    ###    ###       ###                   ##",
        " ##            ::::::::: ::::::::::    :::     :::::::::::::::::::      ##",
        " ##           :+:    :+::+:         :+: :+:  :+:    :+:   :+:           ##",
        " ##          +:+    +:++:+        +:+   +:+ +:+          +:+            ##",
        " ##         +#++:++#+ +#++:++#  +#++:++#++:+#++:++#++   +#+             ##",
        " ##        +#+    +#++#+       +#+     +#+       +#+   +#+              ##",
        " ##       #+#    #+##+#       #+#     #+##+#    #+#   #+#               ##",
        " ##      ######### #############     ### ########    ###                ##",
        " ##                                                                     ##",
        " ##                                                                     ##",
        " #########################################################################",
        " ############################## THE GUI ##################################",
        " #########################################################################"
         
    ]
    for line in text_block:
        terminal.insert(tk.END, line + "\n")

    # Disable the widget so text cannot be edited
    terminal.config(state=tk.DISABLED)
    
    return terminal

def get_prompt():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running get_prompt")

    username = os.getlogin()
    computername = socket.gethostname()
    current_directory = os.getcwd()
    return f"{username}@{computername}:{current_directory}$ "

def select_input_dir():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running select_input_dir")
    
    dir_selected = filedialog.askdirectory(title="Select Input Directory")
    input_dir_var.set(dir_selected)

def select_output_dir():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running select_output_dir")
    
    dir_selected = filedialog.askdirectory(title="Select Output Directory")
    output_dir_var.set(dir_selected)

def select_param_file():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running select_param_file")
    
    pre_specified_directory = baseDir
    file_selected = filedialog.askopenfilename(title="Select Parameter File", initialdir=pre_specified_directory)
    if file_selected.startswith(pre_specified_directory):  # Ensure the file is from the pre-specified directory
        param_file_var.set(os.path.basename(file_selected))
    else:
        print("Please select a file from the specified directory.")

def get_system_path():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running get_system_path")
    
    try:
        result = subprocess.Popen("bash -l -c 'echo $PATH'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = result.communicate()
        return stdout.strip()
    except Exception:
        return None

def read_process_output(process, console):
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running read_process_output")
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            hide_console_input()
            prompt = get_prompt()
            console.insert(tk.END, prompt)
            
            stop_drifting()
            end_drifting_banner()
            break
        if output:
            console.insert(tk.END, output)
            console.see(tk.END)  # Scroll to the end
    process.poll()

def send_input_to_process(process, input_data):
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running send_input_to_process")
    
    process.stdin.write(input_data + '\n')
    process.stdin.flush()

def show_console_input():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running show_console_input")
    
    input_field.pack(fill=tk.X, padx=5, pady=(5, 5), expand=False)


def start_drifting_banner():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running start_drifting_banner")
    
    '''Starts the drifting banner animation.'''
    global drifting_active
    drifting_active = True
    
    canvas.itemconfigure(ready_text, state=tk.HIDDEN)
    canvas.itemconfig(drifting_text, state=tk.NORMAL)
    drift_banner()

def drift_banner():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running drift_banner")
    
    '''Makes the "Job Running..." banner drift left and right.'''
    global drifting_active
    if not drifting_active:
        return
    x, y = canvas.coords(drifting_text)
    direction = 1 if x < canvas.winfo_width() - 20 else -1
    canvas.move(drifting_text, direction, 0)
    root.after(100, drift_banner)

def end_drifting_banner():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running end_drifting_banner")
    
    '''Stops the drifting banner and sets the READY message.'''
    global drifting_active
    drifting_active = False
    canvas.itemconfig(drifting_text, state=tk.HIDDEN)
    canvas.itemconfig(ready_text, state=tk.NORMAL)
    #adjust_ready_position()
    
def update_text_positions(event):
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running update_text_positions")
    
    '''Updates the position of drifting banner and READY text to be centered above console.'''
    x_center = canvas.winfo_width() / 2
    y_position = console.winfo_y() - 20
    canvas.coords(drifting_text, x_center, y_position)
    
    canvas.coords(ready_text, 50, 50)


def run_command():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running run_command")
    
    global console_visible
    
    # Before the command runs, hide the "READY" text and show the drifting text.
    canvas.itemconfig(ready_text, state=tk.HIDDEN)
    canvas.itemconfig(drifting_text, state=tk.NORMAL)
    
    # start the drifting text
    start_drifting()

    if not console_visible:
        #console.pack(fill=tk.BOTH, expand=True)
        console.pack(fill=tk.BOTH, expand=False)
        console_visible = True

    start_drifting_banner()

    show_console_input()  # Always show the input field when running a command

    # Build command from dropdown/button inputs...
    program = program_var.get()
    jobtype = jobtype_var.get()
    command = f"{program} {input_dir_var.get()} {output_dir_var.get()} {jobtype} {param_file_var.get()}"
    console.insert(tk.END, f"$ {command}\n")# start the command with a bash prompt to make it look like a real terminal window..
    system_path = get_system_path() # get the path to add onto the simulated terminal prompt..
    
    try:
        # assemble the rest pf the command, set up process piping details for when command is run/output piped back to console
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
        
        # run the sub-process set up above in the native bash shell
        # on a seperate thread by running the read_process_output() function defined above.
        # read_process_output will run the bash subprocess and handle the displaying of the piped output to the console
        # terminal emulator..
        # 
        thread = threading.Thread(target=read_process_output, args=(process, console))
        thread.daemon = True
        thread.start()

         
        # send user input to actual bash session after they hit henter
        # if input was required/entered in the text input..
        def on_enter_pressed(event):
            global dbMode
            if dbMode == 1:
                global callCountr
                callCountr=callCountr+1
                print(f"Call # {callCountr} : Running on_enter_pressed")
            
            user_input = input_field.get()
            console.insert(tk.END, user_input + '\n')
            send_input_to_process(process, user_input)
            input_field.delete(0, tk.END)

        input_field.bind('<Return>', on_enter_pressed)
    except Exception as e:
        console.insert(tk.END, f"Failed to run command: {e}\n")

def update_canvas_text(*args):
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running update_canvas_text")
    
    canvas.itemconfig(input_dir_label, text=input_dir_var.get())
    canvas.itemconfig(output_dir_label, text=output_dir_var.get())
    canvas.itemconfig(param_file_label, text=param_file_var.get())

def on_canvas_resized(event):
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running on_canvas_resized")
    
    canvas_width = event.width
    canvas_height = event.height
    sections = 6
    
    
    # Terminal Header at Top
    canvas.coords(terminalHeader, canvas_width / 2, canvas_height * 2 / sections)
    
    # Then dropdown for function
    canvas.coords(dropdown_window, (canvas_width / 4)*1, canvas_height * 4.5 / sections)
    canvas.coords(dropdown_label, (canvas_width / 4)*1, canvas_height * 4.5 / sections - 35)
    # Then dropdown for jobtype
    canvas.coords(dropdown_jobtype_window, (canvas_width / 4)*3, canvas_height * 4.5 / sections)
    canvas.coords(jobtype_dropdown_label, (canvas_width / 4)*3, canvas_height * 4.5 / sections - 35)
    
    # Then button for input dir
    canvas.coords(select_input_btn, (canvas_width / 4)*1, canvas_height * 5 / sections)
    canvas.coords(input_dir_label, (canvas_width / 4)*1, canvas_height * 5 / sections + 20)
    # Then button for output dir
    canvas.coords(select_output_btn, (canvas_width / 4)*2, canvas_height * 5 / sections)
    canvas.coords(output_dir_label, (canvas_width / 4)*2, canvas_height * 5 / sections + 20)
    # Then button for par file
    canvas.coords(select_param_btn, (canvas_width / 4)*3, canvas_height * 5 / sections)
    canvas.coords(param_file_label, (canvas_width / 4)*3, canvas_height * 5 / sections + 20)
    
    # Then RUN button
    canvas.coords(run_btn, canvas_width / 2, canvas_height * 5.5 / sections)
    
    
    

    # Adjust the position of the READY and drifting text to be just above the console.
    console_y = canvas_height - console.winfo_height()  # y-coordinate of the top edge of the console
    text_y = console_y - 30  # Positioned 30 pixels above the top edge of the console

    canvas.coords(ready_text, 50, 50)
    
    canvas.coords(drifting_text, canvas_width / 2, text_y)
    
    #adjust_ready_position()
    adjust_drifting_position()

def adjust_ready_position():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running adjust_ready_position")
    
    canvas_width = canvas.winfo_width()
    console_y = canvas.winfo_height() - console.winfo_height()  # y-coordinate of the top edge of the console
    text_y = canvas.winfo_height() - 50  # Positioned 50 pixels from the bottom of the canvas
    canvas.coords(ready_text, 50, 50)

def adjust_drifting_position():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running adjust_drifting_position")
    
    canvas_width = canvas.winfo_width()
    console_y = canvas.winfo_height() - console.winfo_height()  # y-coordinate of the top edge of the console
    text_y = console_y - 30  # Positioned 30 pixels above the top edge of the console
    canvas.coords(drifting_text, canvas_width / 2, text_y)    

def hide_console_input():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running hide_console_input")
    
    input_field.pack_forget()

def drift_text():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running drift_text")
    
    while drift:
        for i in range(0, canvas.winfo_width(), 10):
            if not drift:
                break
            canvas.move(drifting_text, 10, 0)
            root.update()
            root.after(100)

        canvas.coords(drifting_text, -100, canvas.coords(drifting_text)[1])

def start_drifting():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running start_drifting")
    
    global drift
    drift = True
    threading.Thread(target=drift_text, daemon=True).start()

def stop_drifting():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running stop_drifting")
    
    global drift
    drift = False
# ----------------------------------------------------------------------------------------

# root stuff
root = tk.Tk()
root.title("TextyBeast Job Runner")
root.geometry("800x1000")
root.configure(bg="#d9d9d9")
canvas = tk.Canvas(root, bg="#d9d9d9")
#canvas.pack(pady=20, padx=20)

# general canvas stuff?
canvas.pack(fill=tk.BOTH, expand=True)
canvas.bind("<Configure>", on_canvas_resized)

style = ttk.Style()
style.configure("TOptionMenu", background='#ccc')


# Create the terminal banner header and add it to the canvas
terminal = create_terminal(root)
terminalHeader = canvas.create_window(325, 0, anchor=tk.CENTER, window=terminal)


# Choose Function/Program Dropdown
# -------------------------------------------
# make dropdown label 
dropdown_label = canvas.create_text(300, 40, text="Choose a Function:", anchor=tk.N, fill="black", font=("Arial", 10, "bold"))
# default/initial option
program_var = tk.StringVar(value="Select...")
# Set of functions/programs
programs = ["Select...", "textyBeast_localjob", "textyBeast_slurmjob", "textyBeast_remotejob"]
# create dropdown
dropdown = ttk.OptionMenu(root, program_var, *programs)
# create canvas window and set coordinates/set anchor setting
dropdown_window = canvas.create_window(300, 85, window=dropdown, anchor=tk.CENTER)
# -------------------------------------------

# Choose jobtype Dropdown
# -------------------------------------------
# make dropdown label 
jobtype_dropdown_label = canvas.create_text(300, 170, text="Choose a Job Type:", anchor=tk.N, fill="black", font=("Arial", 10, "bold"))
# default/initial option
jobtype_var = tk.StringVar(value="Select...")
# Set of jobtypes
jobtypes = ["Select...", "vl", "di"]
# create dropdown
dropdown_jobtype = ttk.OptionMenu(root, jobtype_var, *jobtypes)
# create canvas window and set coordinates/set anchor setting
dropdown_jobtype_window = canvas.create_window(300, 215, window=dropdown_jobtype, anchor=tk.CENTER)
# -------------------------------------------

# Create labels for the input directory, output directory, and parameter file buttons:
# I think these are the ones that show up/populate below button when selection is made..
# -------------------------------------------
# create string variable for each.. is the "real string" (not label) fed to the run command?
input_dir_var = tk.StringVar()
output_dir_var = tk.StringVar()
param_file_var = tk.StringVar()

# Add "trace"? each of the variables created above.. looks like its calling/involving update_canvas_text funciton for each.. Think it sets up ability to update the _label variables for the labels (below button after pushing) to update to equal whatever was selected for the dir_var variable in the browser/button.. done by update_canvas_text
input_dir_var.trace_add("write", update_canvas_text)
output_dir_var.trace_add("write", update_canvas_text)
param_file_var.trace_add("write", update_canvas_text)

# running canvas.create_text for to create label string for each (below button)
input_dir_label = canvas.create_text(300, 285, text="", anchor=tk.CENTER)
output_dir_label = canvas.create_text(300, 345, text="", anchor=tk.CENTER)
param_file_label = canvas.create_text(300, 405, text="", anchor=tk.CENTER)
# -------------------------------------------

# Create buttons for input dir, output dir, and parameter file selectors..
# make their respective functions called by setting command=function-above so those are called when pressed
# -------------------------------------------
select_input_btn = canvas.create_window(300, 260, window=ttk.Button(canvas, text="Select Input Directory", command=select_input_dir))
select_output_btn = canvas.create_window(300, 320, window=ttk.Button(canvas, text="Select Output Directory", command=select_output_dir))
select_param_btn = canvas.create_window(300, 380, window=ttk.Button(canvas, text="Select Parameter File", command=select_param_file))
# -------------------------------------------

# Create the "Run" Button
# -------------------------------------------
run_btn = canvas.create_window(300, 440, window=ttk.Button(canvas, text="Run", command=run_command))
# -------------------------------------------

# create the console for the terminal emulator...
# -------------------------------------------
console_visible = False
console = scrolledtext.ScrolledText(root, width = 75, height = 20, bg='black', fg='green')
#console.pack(expand=False,side=tk.BOTTOM)
console.pack(expand=False)

#console.pack_forget() # Think this might make it initially "hidden"..?
# -------------------------------------------

# set up the field for the text input to console that appears when job is running...
# -------------------------------------------
input_field = ttk.Entry(root)
input_field.pack_forget() # Think this might make it initially "hidden"..?
# -------------------------------------------

# Drifting banner and READY text coordinates **NEEDED?**
# -------------------------------------------
run_btn_coords = canvas.coords(run_btn)
drifting_text_x = run_btn_coords[0]
drifting_text_y = run_btn_coords[1] - 20
# -------------------------------------------

# Drifting Text Setup
# -------------------------------------------
drifting_text = canvas.create_text(-100, 0, text="Job Running...", fill="red", font=("Arial", 16), state=tk.HIDDEN)
# -------------------------------------------

# "READY" Text Setup
# -------------------------------------------
console_y = canvas.winfo_height() - console.winfo_height()  # y-coordinate of the top edge of the console
text_y = console_y - 30  # Positioned 30 pixels above the top edge of the console
ready_text_x = canvas.winfo_width() / 2
ready_text = canvas.create_text(50, 50, text="READY", fill="green", font=("Arial", 16, "bold"), state=tk.HIDDEN)
# -------------------------------------------

canvas.itemconfig(ready_text, state=tk.NORMAL)

root.mainloop()
