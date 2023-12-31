#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 01:24:56 2022

Started with:
    https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html
    
    **
    https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html#find-the-most-likely-path-backtracking
    **

@author: eduwell
"""

def speech_rec_al(AudioIn, txtFileOut, txtFileOut_al, data_dir, out_dir):
    
    #%% Import Packages
    import os
    from dataclasses import dataclass
    import IPython
    import matplotlib
    import matplotlib.pyplot as plt
    import requests
    import torch
    import torchaudio
    #import moviepy.editor as mp
    #import math
    #import time
    
    #import librosa
    #import matplotlib.pyplot as plt
    #import pandas as pd
    from IPython.display import Audio, display
    import shutil
    import time
    import speech2text as s2t
    #import speech_recognition as sr
    #%% Parameters
    #matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]
    begin_time = time.perf_counter()
    #root_dir = "/Users/eduwell/OneDrive - mcw.edu/duwell/data/EJD_Data_Lab_Projects/Video_Text_Extraction/speech_recognition/"
    root_dir = data_dir
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(torch.__version__)
    print(torchaudio.__version__)
    print(device)
    
    
    #%% Get Data
    os.chdir(root_dir)
    
    
    SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
    SPEECH_FILE = "_assets/speech.wav"
    #SPEECH_FILE = out_dir+'speech.wav'

    if not os.path.exists(SPEECH_FILE):
        os.makedirs("_assets", exist_ok=True)
        with open(SPEECH_FILE, "wb") as file:
            file.write(requests.get(SPEECH_URL).content)
    
    # copy the input file into the directory created/set up above.
    source = data_dir + AudioIn
    destination = out_dir+'speech.wav'
    shutil.copy(source, destination)
    
    #out_dir = data_dir+out_dir
    #%% Run speech_rec to get transcript
    transcript = s2t.speech_rec(AudioIn, txtFileOut, data_dir, out_dir)
    os.chdir(root_dir)
    #%% Generate frame-wise label probability
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    with torch.inference_mode():
        waveform, _ = torchaudio.load(SPEECH_FILE)
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()
    
    # Visualize
    #print(labels)
    #plt.imshow(emission.T)
    #plt.colorbar()
    #plt.title("Frame-wise class probability")
    #plt.xlabel("Time")
    #plt.ylabel("Labels")
    #plt.show()
    
    #%% Generate alignment probability (trellis)
    #transcript = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"
    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [dictionary[c] for c in transcript]
    #print(list(zip(transcript, tokens)))
    
    
    def get_trellis(emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)
    
        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
        trellis[:, 0] = 0
        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis
    
    
    trellis = get_trellis(emission, tokens)
    
    # Visualize
    # plt.imshow(trellis[1:, 1:].T, origin="lower")
    # plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
    # plt.colorbar()
    # plt.show()
    
    #%% Find the most likely path (backtracking)
    @dataclass
    class Point():
        token_index: int
        time_index: int
        score: float
    def backtrack(trellis, emission, tokens, blank_id=0):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()
    
        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
    
            # 2. Store the path with frame-wise probability.
            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))
    
            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]


    path = backtrack(trellis, emission, tokens)
    #print(path)
    
    
    # Visualize 
    def plot_trellis_with_path(trellis, path):
        # To plot trellis with path, we take advantage of 'nan' value
        trellis_with_path = trellis.clone()
        for _, p in enumerate(path):
            trellis_with_path[p.time_index, p.token_index] = float("nan")
        plt.imshow(trellis_with_path[1:, 1:].T, origin="lower")
    
    
    # plot_trellis_with_path(trellis, path)
    # plt.title("The path found by backtracking")
    # plt.show()
    
    # Merge the labels
    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float
    
        def __repr__(self):
            return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"
    
        @property
        def length(self):
            return self.end - self.start
    
    
    def merge_repeats(path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments
    
    
    segments = merge_repeats(path)
    
    #for seg in segments:
    #    print(seg)

    # visualize
    def plot_trellis_with_segments(trellis, segments, transcript):
        # To plot trellis with path, we take advantage of 'nan' value
        trellis_with_path = trellis.clone()
        for i, seg in enumerate(segments):
            if seg.label != "|":
                trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")
    
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
        ax1.set_title("Path, label and probability for each label")
        ax1.imshow(trellis_with_path.T, origin="lower")
        ax1.set_xticks([])
    
        for i, seg in enumerate(segments):
            if seg.label != "|":
                ax1.annotate(seg.label, (seg.start + 0.7, i + 0.3), weight="bold")
                ax1.annotate(f"{seg.score:.2f}", (seg.start - 0.3, i + 4.3))
    
        ax2.set_title("Label probability with and without repetation")
        xs, hs, ws = [], [], []
        for seg in segments:
            if seg.label != "|":
                xs.append((seg.end + seg.start) / 2 + 0.4)
                hs.append(seg.score)
                ws.append(seg.end - seg.start)
                ax2.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
        ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")
    
        xs, hs = [], []
        for p in path:
            label = transcript[p.token_index]
            if label != "|":
                xs.append(p.time_index + 1)
                hs.append(p.score)
    
        ax2.bar(xs, hs, width=0.5, alpha=0.5)
        ax2.axhline(0, color="black")
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(-0.1, 1.1)
    
    
    # plot_trellis_with_segments(trellis, segments, transcript)
    # plt.tight_layout()
    # plt.show()
    
    # Merge words
    def merge_words(segments, separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words
    
    
    word_segments = merge_words(segments)
    
    os.chdir(out_dir)
    with open(txtFileOut_al,mode ='w') as file: 
        for word in word_segments:
            #print(word)
            word_str = str(word)
            file.write(word_str) 
            file.write("\n") 


    

    
    def plot_alignments(trellis, segments, word_segments, waveform):
        trellis_with_path = trellis.clone()
        for i, seg in enumerate(segments):
            if seg.label != "|":
                trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")
    
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
    
        ax1.imshow(trellis_with_path[1:, 1:].T, origin="lower")
        ax1.set_xticks([])
        ax1.set_yticks([])
    
        for word in word_segments:
            ax1.axvline(word.start - 0.5)
            ax1.axvline(word.end - 0.5)
    
        for i, seg in enumerate(segments):
            if seg.label != "|":
                ax1.annotate(seg.label, (seg.start, i + 0.3))
                ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 4), fontsize=8)
    
        # The original waveform
        ratio = waveform.size(0) / (trellis.size(0) - 1)
        ax2.plot(waveform)
        for word in word_segments:
            x0 = ratio * word.start
            x1 = ratio * word.end
            ax2.axvspan(x0, x1, alpha=0.1, color="red")
            ax2.annotate(f"{word.score:.2f}", (x0, 0.8))
    
        for seg in segments:
            if seg.label != "|":
                ax2.annotate(seg.label, (seg.start * ratio, 0.9))
        xticks = ax2.get_xticks()
        plt.xticks(xticks, xticks / bundle.sample_rate)
        ax2.set_xlabel("time [second]")
        ax2.set_yticks([])
        ax2.set_ylim(-1.0, 1.0)
        ax2.set_xlim(0, waveform.size(-1))
    
    
    # plot_alignments(
    #     trellis,
    #     segments,
    #     word_segments,
    #     waveform[0],
    # )
    # plt.show()
    
    
    # A trick to embed the resulting audio to the generated file.
    # `IPython.display.Audio` has to be the last call in a cell,
    # and there should be only one call par cell.
    def display_segment(i):
        ratio = waveform.size(1) / (trellis.size(0) - 1)
        word = word_segments[i]
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        filename = f"_assets/{i}_{word.label}.wav"
        torchaudio.save(filename, waveform[:, x0:x1], bundle.sample_rate)
        print(f"{word.label} ({word.score:.2f}): {x0 / bundle.sample_rate:.3f} - {x1 / bundle.sample_rate:.3f} sec")
        return IPython.display.Audio(filename)


    # with open(txtFileOut_al,mode ='w') as file: 
    #    for word in word_segments:
    #        #print(word)
    #        word_str = str(word)
    #        start_str = str(word.start)
    #        end_str = str(word.end)
    #        word_str2 = word_str+" "+"word_start:"+" "+start_str+" "+"word_end:"+" "+end_str
    #        file.write(word_str2) 
    #        file.write("\n") 


   
    end_time = time.perf_counter()
    time_elapsed = end_time - begin_time
    time_elapsed = str(time_elapsed)
    end_time_message = "Converting this speech audio file to text and alligning the text to the audio took"+" "+time_elapsed+" "+"seconds"
    print(end_time_message)
    