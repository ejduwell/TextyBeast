#import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import torch
import moviepy.editor as mp
import shutil

def run(inFile,OutDir,OutDir_sub,tokenIn,installDir,whspModel,maxSpeakers):
    
    #OutDir='/scratch/g/tark/dataScraping/output'
    #OutDir_sub="test"
    #Preparing the audio file
    # Extract .WAV audio from video and save in audio directory
    #-----------------------------------------------------------------
    #inFile="/scratch/g/tark/dataScraping/envs/gtp/env/input/test_trm.mp4"
    maxSpeakers=int(maxSpeakers);
    clip = mp.VideoFileClip(inFile) 
    audioOut_fname = os.path.splitext(inFile)[0]+".wav" 
    clip.audio.write_audiofile(audioOut_fname)
    clip.audio.to_audiofile(audioOut_fname)
    DEMO_FILE1=audioOut_fname
    #-----------------------------------------------------------------
    
    # Set up video_title and video_id variables based on input file..
    #=========================================================
    # use the os.path.basename() function to get the file name
    file_name = os.path.basename(inFile)
    # Build the descriptive title for the html output page
    video_title="Diarized transcript for: "+file_name
    video_id="videos/"+file_name
    #=========================================================
    
    #Running pyannote.audio to generate the diarizations.
    #=========================================================
    #Go to the output directory
    os.chdir(OutDir)
    
    if not os.path.exists(OutDir_sub):
    	os.mkdir(OutDir_sub)
    
    os.chdir(OutDir_sub)
    os.mkdir("videos")
    # specify the directory where you want to copy the file
    dst_directory = OutDir+"/"+OutDir_sub+"/"+"videos"
    # use the shutil package to copy the file from the source path to the destination directory
    shutil.copy(inFile, dst_directory)
    # Now reassign OutDir to be the output subdir..
    OutDir=OutDir+"/"+OutDir_sub
    
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = AudioSegment.from_wav(DEMO_FILE1) #lecun1.wav
    audio = spacer.append(audio, crossfade=0)
    audio.export('audio.wav', format='wav')
    
    from huggingface_hub import login
    login(tokenIn)
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=True)
    
    DEMO_FILE = {'uri': 'blabla', 'audio': 'audio.wav'}
    dz = pipeline(DEMO_FILE,min_speakers=1, max_speakers=maxSpeakers)  
    #dz = pipeline("audio.wav") 
    
    
    
    with open("diarization.txt", "w") as text_file:
        text_file.write(str(dz))
    
    print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")
    
    #Preparing audio files according to the diarization
    print("Preparing audio files according to the diarization")
    #=========================================================
    def millisec(timeStr):
      spl = timeStr.split(":")
      s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
      return s
    
    #Grouping the diarization segments according to the speaker.
    print("Grouping the diarization segments according to the speaker.")
    import re
    dzs = open('diarization.txt').read().splitlines()
    
    groups = []
    g = []
    lastend = 0
    
    for d in dzs:   
      if g and (g[0].split()[-1] != d.split()[-1]):      #same speaker
        groups.append(g)
        g = []
      
      g.append(d)
      
      end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
      end = millisec(end)
      if (lastend > end):       #segment engulfed by a previous segment
        groups.append(g)
        g = [] 
      else:
        lastend = end
    if g:
      groups.append(g)
    print(*groups, sep='\n')
    
    #Save the audio part corresponding to each diarization group.
    print("Save the audio part corresponding to each diarization group.")
    audio = AudioSegment.from_wav("audio.wav")
    gidx = -1
    for g in groups:
      start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
      end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
      start = millisec(start) #- spacermilli
      end = millisec(end)  #- spacermilli
      print(start, end)
      gidx += 1
      audio[start:end].export(str(gidx) + '.wav', format='wav')
    
    #=========================================================
    
    #Whisper's Transcriptions
    print("Whisper's Transcriptions:")
    #=========================================================
    import datetime
    
    def timedelta_to_videotime(delta):
      """
      Here's a janky way to format a 
      datetime.timedelta to match 
      the format of vtt timecodes. 
      """
      parts = delta.split(":")
      if len(parts[0]) == 1:
        parts[0] = f"0{parts[0]}"
      new_data = ":".join(parts)
      parts2 = new_data.split(".")
      if len(parts2) == 1:
        parts2.append("000")
      elif len(parts2) == 2:
        parts2[1] = parts2[1][:2]
      final_data = ".".join(parts2)
      return final_data
    
    import pathlib
    def whisper_segments_to_vtt_data(result_segments):
    
        """
        This function iterates through all whisper
        segements to format them into WebVTT.
        """
        data = "WEBVTT\n\n"
        for idx, segment in enumerate(result_segments):
            num = idx + 1
            data+= f"{num}\n"
            start_ = datetime.timedelta(seconds=segment.get('start'))
            start_ = timedelta_to_videotime(str(start_))
            end_ = datetime.timedelta(seconds=segment.get('end'))
            end_ = timedelta_to_videotime(str(end_))
            data += f"{start_} --> {end_}\n"
            text = segment.get('text').strip()
            data += f"{text}\n\n"
            return data
    
    import subprocess
    #Run whisper on all audio files. Whisper generates the transcription and writes it to a file.
    os.chdir(OutDir)
    #modelstr="large" #orig
    modelstr=whspModel #for debugging speed..
    dirOut=OutDir
    lang="English"
    print("Begining Whisper Loop..")
    
    afileStr="" #initialize string variable..
    for i in range(gidx+1):   
        afile=OutDir+'/'+str(i)+'.wav'
        afileStr=afileStr+afile+' '
    
    # generate the random "finished" signal for signaling when whisper is done..
    import random
    random.seed()
    finSignal = str(random.randint(10000000, 99999999))+".txt"
    
    #assemble whisper command for whisper environment job..
    whspCmd="cd "+installDir+"; ./whsprXenv_3"+" "+modelstr+" "+dirOut+" "+lang+" "+dirOut+" "+finSignal+" "+afileStr
    #whspCmd="sbatch /scratch/g/tark/installTesting/dataScraping/whsprXenv_2.slurm"+" "+modelstr+" "+dirOut+" "+lang+" "+dirOut+" "+finSignal+" "+afileStr
    subprocess.call(whspCmd, shell=True)
    
    #Generating the HTML file from the Transcriptions and the Diarization
    print("Generating the HTML file from the Transcriptions and the Diarization")
    #=========================================================
    #Change or add to the speaker names and collors bellow as you wish (speaker, textbox color, speaker color).
    speakers = {'SPEAKER_00':('Person 1', 'white', 'darkorange'), 'SPEAKER_01':('Person 2', '#e1ffc7', 'darkgreen'), 'SPEAKER_02':('Person 3', '#e4c7ff', '#8b12fc'), 'SPEAKER_03':('Person 4', '#d9b99a', '#f77d02'), 'SPEAKER_04':('Person 5', '#96a8e0', '#053efa'), 'SPEAKER_05':('Person 6', '#b878bf', '#dc00f5'), 'SPEAKER_06':('Person 7', '#90d1cc', '#019185'), 'SPEAKER_07':('Person 8', '#b6b8b4', '#30302f'), 'SPEAKER_08':('Person 9', '#dbd3a0', '#705d15'), 'SPEAKER_09':('Person 10', '#5eb5b2', '#043331')}
    def_boxclr = 'white'
    def_spkrclr = 'orange'
    
    #In the generated HTML, the transcriptions for each diarization group are written in a box, with the speaker name on the top. By clicking a transcription, the embedded video jumps to the right time .
    #preS = '<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <meta http-equiv="X-UA-Compatible" content="ie=edge">\n    <title>' + \
    #      video_title + \
    #      '</title>\n    <style>\n        body {\n            font-family: sans-serif;\n            font-size: 18px;\n            color: #111;\n            padding: 0 0 1em 0;\n\t        background-color: #efe7dd;\n        }\n        table {\n             border-spacing: 10px;\n        }\n        th { text-align: left;}\n        .lt {\n          color: inherit;\n          text-decoration: inherit;\n        }\n        .l {\n          color: #050;\n        }\n        .s {\n            display: inline-block;\n        }\n        .c {\n            display: inline-block;\n        }\n        .e {\n            /*background-color: white; Changing background color */\n            border-radius: 20px; /* Making border radius */\n            width: fit-content; /* Making auto-sizable width */\n            height: fit-content; /* Making auto-sizable height */\n            padding: 5px 30px 5px 30px; /* Making space around letters */\n            font-size: 18px; /* Changing font size */\n            display: flex;\n            flex-direction: column;\n            margin-bottom: 10px;\n            white-space: nowrap;\n        }\n\n        .t {\n            display: inline-block;\n        }\n        #player {\n            position: sticky;\n            top: 20px;\n            float: right;\n        }\n    </style>\n\t<script>\n      var tag = document.createElement(\'script\');\n      tag.src = "https://www.youtube.com/iframe_api";\n      var firstScriptTag = document.getElementsByTagName(\'script\')[0];\n      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);\n      var player;\n      function onYouTubeIframeAPIReady() {\n        player = new YT.Player(\'player\', {\n          //height: \'210\',\n          //width: \'340\',\n          videoId: \'' + \
    #      video_id + \
    #      '\',\n        });\n      }\n      function jumptoTime(timepoint, id) {\n        event.preventDefault();\n        history.pushState(null, null, "#"+id);\n        player.seekTo(timepoint);\n        player.playVideo();\n      }\n    </script>\n  </head>\n  <body>\n    <h2>' + \
    #      video_title + \
    #      '</h2>\n  <i>Click on a part of the transcription, to jump to its video, and get an anchor to it in the address bar<br><br></i>\n<div  id="player"></div>\n'
    #postS = '\t</body>\n</html>'
    #
    #preS = '<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <meta http-equiv="X-UA-Compatible" content="ie=edge">\n    <title>' + \
    #      video_title + \
    #      '</title>\n    <style>\n        body {\n            font-family: sans-serif;\n            font-size: 18px;\n            color: #111;\n            padding: 0 0 1em 0;\n\t        background-color: #efe7dd;\n        }\n        table {\n             border-spacing: 10px;\n        }\n        th { text-align: left;}\n        .lt {\n          color: inherit;\n          text-decoration: inherit;\n        }\n        .l {\n          color: #050;\n        }\n        .s {\n            display: inline-block;\n        }\n        .c {\n            display: inline-block;\n        }\n        .e {\n            /*background-color: white; Changing background color */\n            border-radius: 20px; /* Making border radius */\n            width: fit-content; /* Making auto-sizable width */\n            height: fit-content; /* Making auto-sizable height */\n            padding: 5px 30px 5px 30px; /* Making space around letters */\n            font-size: 18px; /* Changing font size */\n            display: flex;\n            flex-direction: column;\n            margin-bottom: 10px;\n            white-space: nowrap;\n        }\n\n        .t {\n            display: inline-block;\n        }\n        #player {\n            position: sticky;\n            top: 20px;\n            float: right;\n        }\n    </style>\n\t<script>\n      var tag = document.createElement(\'script\');\n      tag.src = "https://www.youtube.com/iframe_api";\n      var firstScriptTag = document.getElementsByTagName(\'script\')[0];\n      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);\n      var player;\n      function onYouTubeIframeAPIReady() {\n        player = new YT.Player(\'player\', {\n          //height: \'210\',\n          //width: \'340\',\n          videoId: \'' + \
    #      video_id + \
    #      '\',\n        });\n      }\n      function jumptoTime(timepoint, id) {\n        event.preventDefault();\n        history.pushState(null, null, "#"+id);\n        player.seekTo(timepoint);\n        player.playVideo();\n      }\n    </script>\n  </head>\n  <body>\n    <h2>' + \
    #      video_title + \
    #      '</h2>\n  <i>Click on a part of the transcription, to jump to its video, and get an anchor to it in the address bar<br><br></i>\n<div  id="player"></div>\n'
    #postS = '\t</body>\n</html>'
    
    
    
    # ETHANS ATTEMPT AT ADAPTING HTML GENERATION TO MATCH ADAPTED HTML PAGE VERSION HE GOT WORKING LOCALLY..
    preS = '<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <meta http-equiv="X-UA-Compatible" content="ie=edge">\n    <link href="https://vjs.zencdn.net/7.14.3/video-js.css" rel="stylesheet" />\n    <script src="https://vjs.zencdn.net/7.14.3/video.js"></script>\n    <title>'+ \
            video_title + \
            '</title>\n    <style>\n        body {\n            font-family: sans-serif;\n            font-size: 18px;\n            color: #111;\n            padding: 0 0 1em 0;\n\t        background-color: #ADD8E6;\n        }\n        table {\n             border-spacing: 10px;\n        }\n        th { text-align: left;}\n        .lt {\n          color: inherit;\n          text-decoration: inherit;\n        }\n        .l {\n          color: #050;\n        }\n        .s {\n            display: inline-block;\n        }\n        .c {\n            display: inline-block;\n        }\n        .e {\n            /*background-color: white; Changing background color */\n            border-radius: 20px; /* Making border radius */\n            width: fit-content; /* Making auto-sizable width */\n            height: fit-content; /* Making auto-sizable height */\n            padding: 5px 30px 5px 30px; /* Making space around letters */\n            font-size: 18px; /* Changing font size */\n            display: flex;\n            flex-direction: column;\n            margin-bottom: 10px;\n            white-space: nowrap;\n        }\n\n\t.z {\n\t    /*-- Formatting for Title banner at top of page...  */\n           \n            border-radius: 20px; /* Making border radius */\n \n            height: fit-content; /* Making auto-sizable height */\n            padding: 5px 30px 5px 30px; /* Making space around letters */\n            font-size: 18px; /* Changing font size */\n            display: flex;\n            flex-direction: column;\n            margin-bottom: 10px;\n            white-space: nowrap;\n        }\n\t.o {\n            /*background-color: white; Changing background color */\n            border-radius: 20px; /* Making border radius */\n            width: fit-content; /* Making auto-sizable width */\n            height: fit-content; /* Making auto-sizable height */\n            padding: 5px 30px 5px 30px; /* Making space around letters */\n            font-size: 18px; /* Changing font size */\n            display: flex;\n            flex-direction: column;\n            margin-bottom: 10px;\n            white-space: nowrap;\n        }\n\n        .t {\n            display: inline-block;\n        }\n        #player {\n            position: sticky;\n            top: 20px;\n            float: right;\n\t    border-radius: 10px;\n        }\n    </style>\n\n    <script>\n     var player;\n\n      function jumptoTime(timepoint, id) {\n        event.preventDefault();\n        history.pushState(null, null, "#" + id);\n\tif (!isNaN(timepoint)) {\n            player.currentTime(timepoint);\n\t}\n        player.play();\n      }\n\n      document.addEventListener(\'DOMContentLoaded\', function() {\n        player = videojs(\'player\', {\n          controls: true,\n          autoplay: true,\n          sources: [{\n            src: \''+ \
            video_id +"'" \
            ',\n            type: \'video/mp4\'\n          }]\n        });\n\n        player.ready(function() {\n          var transcript = document.querySelector(\'.transcript\');\n          var captions = transcript.querySelectorAll(\'a\');\n          for (var i = 0; i < captions.length; i++) {\n            var caption = captions[i];\n            caption.addEventListener(\'click\', function(e) {\n              jumptoTime(parseFloat(this.dataset.start), this.id);\n            });\n          }\n        });\n      });\n      \n</script>\n  </head>\n  <body>\n    <div style="text-align: center;">\n    <div class="z" style="background-color: #f5f5f5"; padding: 20px; box-shadow: \n    0 0 5px 5px rgba(0, 0, 0, 0.5), \n    0 0 0 10px #F5F5F5;">\n    <h2 style="color: blue;">'+\
            video_title + \
            '</h2>\n    <i>Click on a part of the transcription, to jump to that location in the video, and get an anchor to it in the address bar<br><br></i>\n    </div>\n    </div>\n  <video id="player" class="video-js vjs-default-skin" controls preload="auto" width="640" height="264"\n      data-setup=\'{}\'>\n      <source src= "' + \
            video_id +'"' \
            ' type="video/mp4" />\n      <p class="vjs-no-js">\n        To view this video please enable JavaScript, and consider upgrading to a web browser that supports HTML5 video\n      </p>\n   </video>\n\n<div class="transcript">\n' 
    postS = '\t</body>\n</html>'
    
    import webvtt
    
    from datetime import timedelta
    txtOut=[];
    html = list(preS)
    gidx = -1
    for g in groups:  
      shift = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
      shift = millisec(shift) - spacermilli #the start time in the original video
      shift=max(shift, 0)
      
      gidx += 1
      captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read(str(gidx) + '.wav.vtt')]
      
      print(captions)
      
      if captions:
        speaker = g[0].split()[-1]
        boxclr = def_boxclr
        spkrclr = def_spkrclr
        if speaker in speakers:
          speaker, boxclr, spkrclr = speakers[speaker] 
        html.append(f'<div class="e" style="background-color: {boxclr}">\n');
        html.append(f'<span style="color: {spkrclr}">{speaker}</span><br>\n')
        
        for c in captions:
          start = shift + c[0] 
          end = shift + c[1] 
    
          start = start / 1000   #time resolution ot youtube is Second.
          end = end / 1000   #time resolution ot youtube is Second.
          
          startStr = '{0:02d}:{1:02d}:{2:02.2f}'.format((int)(start // 3600), 
                                                  (int)(start % 3600 // 60), 
                                                  start % 60)      
          endStr = '{0:02d}:{1:02d}:{2:02.2f}'.format((int)(end // 3600), 
                                                  (int)(end % 3600 // 60), 
                                                  end % 60)    
          #html.append(f'<div class="c">')
          #html.append(f'\t\t\t\t<a class="l" href="#{startStr}" id="{startStr}">#</a> \n')
          html.append(f'\t\t\t\t<a href="#{startStr}" id="{startStr}" class="lt" onclick="jumptoTime({int(start)}, this.id)">{c[2]}</a>\n')
          txtOut.append("\n"+speaker+":"+"\n")
          strTmp=startStr+" --> "+endStr+"\n"
          txtOut.append(strTmp)
          txtOut.append(c[2])
          #html.append(f'\t\t\t\t<div class="t"> {c[2]}</div><br>\n')
          #html.append(f'</div>')
        txtOut.append("\n")
        html.append(f'</div>\n');
    
    html.append(postS)
    s = "".join(html)
    t = "".join(txtOut)
    with open("capspeaker.html", "w") as html_file:
        html_file.write(s)
    html_file.close()
    
    with open("diarized_transcript.txt", "w") as text_file:
        text_file.write(t)
    text_file.close()
    print("")
    print("HTML FILE:")
    print(s)
    print("")
    print("TEXT FILE:")
    print(t)
