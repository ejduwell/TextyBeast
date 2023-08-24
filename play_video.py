# play_video.py

import sys
import pygame
from moviepy.editor import VideoFileClip



def play_video(video_path):
    pygame.init()
    clip = VideoFileClip(video_path)
    width, height = clip.size
    screen = pygame.display.set_mode((width, height))
    clip.preview(fps=24, audio=True, audio_fps=44100)
    pygame.quit()


video_path = sys.argv[1]  # Get the video path from the command-line argument
play_video(video_path)
