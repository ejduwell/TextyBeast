#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import subprocess
import threading
import os
import socket
import sys
import signal
from pytube import YouTube
import random
import shutil
import glob

# Get input args
# ----------------------------------------------------------------------------------------
baseDir = sys.argv[1];
clstrPath = sys.argv[2];
clstrUsr= sys.argv[3];
clstrHost = sys.argv[4];

#Build Command to export path/username/hostname info for cluster..
# ----------------------------------------------------------------------------------------
global xprtCmd
xprtCmd= "export clstrPath="+clstrPath+"; "+"export clstrUsr="+clstrUsr+"; "+"export clstrHost="+clstrHost+"; "
print("")
print("This is xprtCmd:")
print(xprtCmd)
print("")

# Set Global Variables:
# ----------------------------------------------------------------------------------------
callCountr=0 # initialize for counting calls in dbMode..
drifting_active = False # global variable to control the drifting animation loop
dbMode=1 # if = 1: turnos on debug mode to display call db printouts on command line...
global process
process = None
global killProcess
killProcess = 0
global exit_thread_flag
exit_thread_flag = threading.Event()
# ----------------------------------------------------------------------------------------
global videoPath
videoPath='/Users/eduwell/GitHub/videos/eduwellDissertationClip.mp4'
ytURLs = ['https://www.youtube.com/watch?v=scD4_ZVDD-8&ab_channel=MontyPython',
          'https://www.youtube.com/watch?v=l9SqQNgDrgg&ab_channel=MontyPython',
          'https://www.youtube.com/watch?v=ohDB5gbtaEQ&ab_channel=unmusedtails',
          'https://www.youtube.com/watch?v=LFrdqQZ8FFc&ab_channel=TylerBird',
          'https://www.youtube.com/watch?v=ycKNt0MhTkk&ab_channel=Weidmoo',
          'https://www.youtube.com/watch?v=B3KBuQHHKx0&ab_channel=It%27sSam%21',
          'https://www.youtube.com/watch?v=imhrDrE4-mI&list=PLhboEWcuNB1__DiH8Z5MLYigS-KVxWktS&ab_channel=KevinSharrock',
          'https://www.youtube.com/watch?v=vZw35VUBdzo&list=PLhboEWcuNB1__DiH8Z5MLYigS-KVxWktS&index=5&ab_channel=Cinematheia',
          'https://www.youtube.com/watch?v=t2c-X8HiBng&list=PLhboEWcuNB1__DiH8Z5MLYigS-KVxWktS&index=17&ab_channel=Browningate',
          'https://www.youtube.com/watch?v=qgSzGIkFq2A&list=PLhboEWcuNB1__DiH8Z5MLYigS-KVxWktS&index=28&ab_channel=NavajoNIJ',
          'https://www.youtube.com/watch?v=SJUhlRoBL8M&list=PLhboEWcuNB1__DiH8Z5MLYigS-KVxWktS&index=40&ab_channel=Melonhead622',
          'https://www.youtube.com/watch?v=pfRdur8GLBM&list=PLhboEWcuNB1__DiH8Z5MLYigS-KVxWktS&index=7&ab_channel=ChristiaanJanssens',
          'https://www.youtube.com/watch?v=0e2kaQqxmQ0&list=PLhboEWcuNB1__DiH8Z5MLYigS-KVxWktS&index=49&ab_channel=TheFilmClipBaron',
          'https://www.youtube.com/watch?v=Ut116mBuPpg',
          'https://www.youtube.com/watch?v=XbY8MH1TpEw',
          'https://www.youtube.com/watch?v=Xm08nJ4opac',
          'https://www.youtube.com/watch?v=ZZ5LpwO-An4',
          'https://www.youtube.com/watch?v=U9t-slLl30E',
          'https://www.youtube.com/watch?v=KaqC5FnvAEc',
          'https://www.youtube.com/watch?v=oavMtUWDBTM',
          'https://www.youtube.com/watch?v=Tx1XIm6q4r4',
          'https://www.youtube.com/watch?v=uE-1RPDqJAY',
          'https://www.youtube.com/watch?v=BBGEG21CGo0',
          'https://www.youtube.com/watch?v=QH2-TGUlwu4',
          'https://www.youtube.com/watch?v=0RpdPzJgaBw',
          'https://www.youtube.com/watch?v=u8ccGjar4Es',
          'https://www.youtube.com/watch?v=cGc_NfiTxng',
          'https://www.youtube.com/watch?v=jofNR_WkoCE',
          'https://www.youtube.com/watch?v=HHFuTpVvRCI',
          'https://www.youtube.com/watch?v=HhGuXCuDb1U',
          'https://www.youtube.com/watch?v=ePsW0wEtKt8&list=PLdcjoWiPNS3MhBeEt5qs6Akcwg8810Rx-&index=7',
          'https://www.youtube.com/watch?v=yoEezZD71sc&list=PLdcjoWiPNS3MyRksQQX__hpicPXjSfnn6',
          'https://www.youtube.com/watch?v=kr1I3mBojc0',
          'https://www.youtube.com/watch?v=CsGYh8AacgY',
          'https://www.youtube.com/watch?v=ExukCRD7gN0',
          'https://www.youtube.com/watch?v=djK_ucSYpaw',
          'https://www.youtube.com/shorts/QQNL83fhWJU',
          'https://www.youtube.com/watch?v=LDU_Txk06tM',
          'https://www.youtube.com/watch?v=EQ1HKCYJM5U',
          'https://www.youtube.com/watch?v=0EqSXDwTq6U',
          'https://www.youtube.com/watch?v=o0u4M6vppCI',
          'https://www.youtube.com/watch?v=8Gv0H-vPoDc',
          'https://www.youtube.com/watch?v=-UYgORr5Qhg',
          'https://www.youtube.com/watch?v=J---aiyznGQ',
          'https://www.youtube.com/watch?v=yBLdQ1a4-JI',
          'https://www.youtube.com/watch?v=EIyixC9NsLI',
          'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
          'https://www.youtube.com/watch?v=mfhBM_Yay6w',
          'https://www.youtube.com/watch?v=HrzY_HF-Znw']


tmpPath=baseDir+"/tmp"

isExist = os.path.exists(tmpPath)
if not isExist:
   # Create a new directory because it does not exist
   os.mkdir(tmpPath)

# Define Functions
# ----------------------------------------------------------------------------------------
def get_random_file(dir_path):
    """
    Get a random file from the specified directory.
    
    Args:
    - dir_path (str): Path to the directory.
    
    Returns:
    - str: Path to the random file.
    """
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    if not files:  # No files in directory
        return None
    return os.path.join(dir_path, random.choice(files))

def getYTvideo(ytURL):
    global baseDir
    global videoPath
    global tmpPath

    subprocess.run(["python", baseDir+"/envs/gui/env/lib/python3.9/site-packages/"+"ytDL.py", ytURL, tmpPath])

 
def play_video():
    global baseDir
    global videoPath
    global tmpPath
    global ytURLs

    

    while True:  # Infinite loop, we will break out when we succeed.
       url = random.choice(ytURLs)
       getYTvideo(url)
       # grab latest file (one we just downloaded..)
       list_of_files = glob.glob(tmpPath + '/*')  # * means all if need specific format then *.csv
       try:
          latest_file = max(list_of_files, key=os.path.getctime)
          break

       except:
          print("")
          print("Error Trying To Get Video..")
          print("The Problem URL was:"+url)
          print("Trying Another URL ...")
          print("")
          continue
        
    video_path = latest_file
    try:
        process = subprocess.Popen(["python", baseDir + "/envs/gui/env/lib/python3.9/site-packages/" + "play_video.py", video_path])
        process.communicate()  # Wait for the process to finish
        if process.returncode != 0:
            print(f"Video player exited with code {process.returncode}.")
    except Exception as e:
        print("Error while playing the video:", str(e))
    os.remove(video_path)

    
def open_amusement_portal():
    amusement_window = tk.Toplevel(root)
    amusement_window.title("Amusement Portal")
    amusement_window.geometry("640x480")

    # General Color Stuff:
    bgClr="#091c30"
    btnClr="#292b45"
    btnClrActive="#07192e"
    amusement_window.configure(bg=bgClr)
    canvas2 = tk.Canvas(amusement_window, bg=bgClr)
    
    # Configure ttk styles
    # -------------------------------------------------------------------
    style = ttk.Style()
    style.theme_use('clam')

    # normal text buttons:
    # Disable the default button focus background change
    style.configure('Custom.TButton',
                    background=btnClr,
                    foreground='#1bde42',
                    borderwidth=4,
                    padding=2,
                    relief='raised',
                    highlightbackground='red')

    style.map('Custom.TButton',
              background=[('active', btnClrActive),
                          ('!active', btnClr)],
              foreground=[('active', '#d959b5'),
                          ('!active', '#1bde42')],
              )
    play_vid_button = ttk.Button(amusement_window, text="Play a Random Silly Video ...", command=play_video, style="Custom.TButton")
    play_vid_button.pack(pady=10)

    close_button = ttk.Button(amusement_window, text="I am now sufficiently amused. Please close this window ...", command=amusement_window.destroy, style="Custom.TButton")
    close_button.pack(pady=10)
    
def close_video_window(video_window):
    video_window.destroy()


def close_window():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running close_window")
    
    # Create a sentinel file
    with open("terminate_sentinel.txt", "w") as f:
        f.write("Terminate without restart.")

    # delete the tmp dir...
    shutil.rmtree(tmpPath)

    # Exit the program
    root.quit()


def kill_process():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running kill_process")
    
    global process, exit_thread_flag
    if process:
        try:
                
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.kill()
            console.insert(tk.END, "Process terminated\n")
            
            #process.terminate()
            #process.kill()
            # Signal the thread to exit
            exit_thread_flag.set()
            hide_console_input()
            prompt = get_prompt()
            console.insert(tk.END, prompt)
            stop_drifting()
            end_drifting_banner()
            
        except Exception as e:
            console.insert(tk.END, f"Failed to terminate process: {e}\n")

    
def allow_highlight(event):
    event.widget.config(state=tk.NORMAL)
    event.widget.after(100, lambda: event.widget.config(state=tk.DISABLED))
    
def configure_dropdown_menu(dropdown, bg_color, fg_color, activebg, activefg):
    # Configure the dropdown menu appearance
    dropdown['menu'].config(bg=bg_color,fg=fg_color, activebackground=activebg, activeforeground=activefg)
    
    for item in dropdown['menu'].winfo_children():
        item.configure(background=bg_color, foreground=fg_color, activebackground=activebg, activeforeground=activefg)

    return dropdown

def create_terminal(canvas):
    terminal = tk.Text(canvas, bg="#07192e", fg="#1bde42", font=("Courier", 10), width=75, height=22)
    
    # Change the highlighted text color and background
    terminal.tag_configure("sel", foreground="#07192e", background="#d959b5")
    
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

    for line in text_block:
        terminal.insert(tk.END, line + "\n")

    terminal.bind("<B1-Motion>", allow_highlight)

    # Disable the widget so text cannot be edited
    #terminal.config(state=tk.DISABLED)
    
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
        result = subprocess.Popen("exec " +"bash -l -c 'echo $PATH'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = result.communicate()
        return stdout.strip()
    except Exception:
        return None

def read_process_output(process, console):
    global exit_thread_flag
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
        
        #global killProcess
        #if killProcess==1:
        #    end_drifting_banner()
        #    hide_console_input()
        #    stop_drifting()
        #    drifting_active = False    
        # Check if the exit flag is set
        if exit_thread_flag.is_set():
            break
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
    


def run_command():
    global dbMode
    global exit_thread_flag
    global xprtCmd
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
        bash_command = xprtCmd+bash_command
        
        #env = {"PATH": system_path} if system_path else None
        
        env = os.environ.copy()  # Start with the current environment
        if system_path:
            env["PATH"] = system_path
        
        global process
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
        exit_thread_flag.clear()
        thread = threading.Thread(target=read_process_output, args=(process, console))
        thread.daemon = True
        thread.start()
        #thread.run()
         
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
    
    
    text_ipd = input_dir_var.get()
    if len(text_ipd) > 20:
        text_ipd = "..."+text_ipd[-20:]
    
    text_opd = output_dir_var.get()
    if len(text_opd) > 20:
        text_opd = "..."+text_opd[-20:]
    
    text_pf = param_file_var.get()
    if len(text_pf) > 20:
        text_pf = "..."+text_ipd[-20:]
    
    canvas.itemconfig(input_dir_label, text=text_ipd)
    canvas.itemconfig(output_dir_label, text=text_opd)
    canvas.itemconfig(param_file_label, text=text_pf)

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
    canvas.coords(terminalHeader, canvas_width / 2, canvas_height * 2.2 / sections)
    
    # The Drifting "Job Running" text...
    #canvas.coords(drifting_text,canvas_width / 2, 10)
    canvas.coords(drifting_text, canvas_width / 2, canvas_height * 1 / sections - 25)

    # Ready Text..
    #canvas.coords(ready_text, canvas_width / 2, 10)console
    canvas.coords(ready_text, canvas_width / 2, canvas_height * 1 / sections - 25)
    
    
    
    # Then dropdown for function
    canvas.coords(dropdown_window, (canvas_width / 4)*1, canvas_height * 4.25 / sections)
    canvas.coords(dropdown_label, (canvas_width / 4)*1, canvas_height * 4.25 / sections - 40)
    # Then dropdown for jobtype
    canvas.coords(dropdown_jobtype_window, (canvas_width / 4)*3, canvas_height * 4.25 / sections)
    canvas.coords(jobtype_dropdown_label, (canvas_width / 4)*3, canvas_height * 4.25 / sections - 40)
    
    # Then button for input dir
    canvas.coords(select_input_btn, (canvas_width / 4)*1, canvas_height * 4.75 / sections)
    canvas.coords(input_dir_label, (canvas_width / 4)*1, canvas_height * 4.75 / sections + 25)
    # Then button for output dir
    canvas.coords(select_output_btn, (canvas_width / 4)*2, canvas_height * 4.75 / sections)
    canvas.coords(output_dir_label, (canvas_width / 4)*2, canvas_height * 4.75 / sections + 25)
    # Then button for par file
    canvas.coords(select_param_btn, (canvas_width / 4)*3, canvas_height * 4.75 / sections)
    canvas.coords(param_file_label, (canvas_width / 4)*3, canvas_height * 4.75 / sections + 25)
    
    # Then RUN button
    canvas.coords(run_btn, canvas_width / 2, canvas_height * 5.35 / sections - 5)
    
    # The Kill Button
    canvas.coords(kill_button, canvas_width / 2, canvas_height * 5.35 / sections +30)
    
    # The Close Button
    canvas.coords(close_button, 75, 25)


    canvas.coords(open_btn,canvas_width - 75,25)
    
    # Adjust the position of the READY and drifting text to be just above the console.
    #console_y = canvas_height - console.winfo_height()  # y-coordinate of the top edge of the console
    #text_y = console_y - 30  # Positioned 30 pixels above the top edge of the console

    #canvas.coords(ready_text, canvas_width / 2, 50)
    
    #canvas.coords(drifting_text, canvas_width / 2, text_y)
    
    #adjust_ready_position()
    #adjust_drifting_position()

def adjust_ready_position():
    global dbMode
    if dbMode == 1:
        global callCountr
        callCountr=callCountr+1
        print(f"Call # {callCountr} : Running adjust_ready_position")
    
    canvas_width = canvas.winfo_width()
    console_y = canvas.winfo_height() - console.winfo_height()  # y-coordinate of the top edge of the console
    text_y = canvas.winfo_height() - 50  # Positioned 50 pixels from the bottom of the canvas
    canvas.coords(ready_text, 50, 25)

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

# General Color Stuff:
bgClr="#091c30"
btnClr="#292b45"
btnClrActive="#07192e"

primTxtClr="#1bde42"
secTxtClr="#d959b5"

# root stuff
root = tk.Tk()
root.title("TextyBeast Job Runner")
root.geometry("800x1000")
root.configure(bg=bgClr)
canvas = tk.Canvas(root, bg=bgClr)
#canvas.pack(pady=20, padx=20)

# general canvas stuff?
canvas.pack(fill=tk.BOTH, expand=True)
canvas.bind("<Configure>", on_canvas_resized)

#style = ttk.Style()
#style.configure("TOptionMenu", background=btnClr)


# Create the terminal banner header and add it to the canvas
terminal = create_terminal(root)
terminalHeader = canvas.create_window(325, 0, anchor=tk.CENTER, window=terminal)


# Choose Function/Program Dropdown
# make dropdown label 
dropdown_label = canvas.create_text(300, 40, text="Choose a Function:", anchor=tk.N, fill="#1bde42", font=("Courier", 12, "bold"))

# Set the theme to clam
style = ttk.Style()
style.theme_use('clam')

# Style for the actual OptionMenu button
style.configure('Custom.TMenubutton',
                background=btnClr,
                foreground=primTxtClr,
                borderwidth=4,
                padding=2,
                highlightthickness=4,
                highlightbackground=secTxtClr,
                relief='raised')

style.map('Custom.TMenubutton',
          background=[('active', btnClrActive),
                      ('!active', btnClr),
                      ('focus', btnClrActive),
                      ('!focus', btnClr)],
          foreground=[('active', secTxtClr),
                      ('!active', primTxtClr),
                      ('focus', secTxtClr),
                      ('!focus',primTxtClr)],
          )

# Create a StringVar for the dropdown
program_var = tk.StringVar(value="Select...")
programs = ["Select...", "textyBeast_localjob", "textyBeast_slurmjob", "textyBeast_remotejob"]

# Create a dropdown using OptionMenu
dropdown = ttk.OptionMenu(canvas, program_var, *programs, style="Custom.TMenubutton")

dropdown.pack(padx=4, pady=4)  # This padding simulates the border width

# Use the frame (with the dropdown inside it) as the window for the canvas
dropdown_window = canvas.create_window(300, 85, window=dropdown, anchor=tk.CENTER)

# -------------------------------------------

# Choose jobtype Dropdown
# -------------------------------------------
# make dropdown label 
jobtype_dropdown_label = canvas.create_text(300, 170, text="Choose a Job Type:", anchor=tk.N, fill="#1bde42", font=("Courier", 12, "bold"))


# Create a StringVar for the dropdown
jobtype_var = tk.StringVar(value="Select...")
# Set of jobtypes
jobtypes = ["Select...", "vl", "di"]

# Create a dropdown using OptionMenu
dropdown_jobtype = ttk.OptionMenu(canvas, jobtype_var, *jobtypes, style="Custom.TMenubutton")

dropdown_jobtype.pack(padx=4, pady=4)  # This padding simulates the border width

# Use the frame (with the dropdown inside it) as the window for the canvas
dropdown_jobtype_window = canvas.create_window(300, 85, window=dropdown_jobtype, anchor=tk.CENTER)

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
input_dir_label = canvas.create_text(300, 285, text="", anchor=tk.CENTER, fill="#d959b5", font=("Courier", 10))
output_dir_label = canvas.create_text(300, 345, text="", anchor=tk.CENTER, fill="#d959b5", font=("Courier", 10))
param_file_label = canvas.create_text(300, 405, text="", anchor=tk.CENTER, fill="#d959b5", font=("Courier", 10))
# -------------------------------------------

# Create buttons for input dir, output dir, and parameter file selectors..
# make their respective functions called by setting command=function-above so those are called when pressed
# -------------------------------------------

# Configure ttk styles
# -------------------------------------------------------------------
style = ttk.Style()
style.theme_use('clam')

# normal text buttons:
# Disable the default button focus background change
style.configure('Custom.TButton',
                background=btnClr,
                foreground='#1bde42',
                borderwidth=4,
                padding=2,
                relief='raised',
                highlightbackground='red')

style.map('Custom.TButton',
          background=[('active', btnClrActive),
                      ('!active', btnClr)],
          foreground=[('active', '#d959b5'),
                      ('!active', '#1bde42')],
          )

# Red text buttons:
style.configure('RedCustom.TButton',
                background=btnClr,
                foreground='#ff0000',  # Red color
                borderwidth=4,
                padding=2,
                relief='raised',
                highlightbackground='red')

style.map('RedCustom.TButton',
          background=[('active', btnClrActive),
                      ('!active', btnClr)],
          foreground=[('active', '#d959b5'),
                      ('!active', '#ff0000')]  # Red color
          )

# -------------------------------------------------------------------

# Select input dir button:
button = ttk.Button(canvas, text="Select Input Directory", command=select_input_dir, style='Custom.TButton')
select_input_btn = canvas.create_window(300, 260, window=button)

# Select output dir button:
button = ttk.Button(canvas, text="Select Output Directory", command=select_output_dir, style='Custom.TButton')
select_output_btn = canvas.create_window(300, 320, window=button)

# Select parameter file button:
button = ttk.Button(canvas, text="Select Parameter File", command=select_param_file, style='Custom.TButton')
select_param_btn = canvas.create_window(300, 380, window=button)

# Run Button:
button = ttk.Button(canvas, text="RUN", command=run_command, style='Custom.TButton')
run_btn = canvas.create_window(300, 440, window=button)
# -------------------------------------------

# create the console for the terminal emulator...
# -------------------------------------------
console_visible = False
console = scrolledtext.ScrolledText(root, width = 75, height = 20, bg='#07192e', fg='#1bde42')
#console.pack(expand=False,side=tk.BOTTOM)
console.pack(expand=False, fill=tk.BOTH)
# -------------------------------------------

# set up the field for the text input to console that appears when job is running...
# -------------------------------------------
input_field = tk.Entry(root, bg='#d959b5', fg='#07192e', insertbackground='#1bde42',insertwidth=10)
input_field.pack_forget() # Think this might make it initially "hidden"..?
# -------------------------------------------

# Drifting Text Setup
# -------------------------------------------
drifting_text = canvas.create_text(-100, 0, text="JOB RUNNING...", fill="#d959b5", font=("Courier", 16, "bold"), state=tk.HIDDEN)
# -------------------------------------------

# "READY" Text Setup
# -------------------------------------------
ready_text = canvas.create_text(50, 20, text="----- READY -----", fill="#1bde42", font=("Courier", 16, "bold"), state=tk.NORMAL)
# -------------------------------------------

# Kill Button
button = ttk.Button(canvas, text="Kill Current Process...", command=kill_process, style='RedCustom.TButton')
kill_button = canvas.create_window(300, 260, window=button)

# Close Button
button = ttk.Button(canvas, text="CLOSE", command=close_window, style='RedCustom.TButton')
close_button = canvas.create_window(50, 50, window=button)


button = ttk.Button(root, text="Amuse Me ...", command=open_amusement_portal, style='Custom.TButton')
button.pack(pady=4)
open_btn = canvas.create_window(300, 380, window=button)

root.mainloop()
