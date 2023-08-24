from pytube import YouTube
import sys

def download_video(url, save_path='.'):
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()  # You can filter streams based on your criteria
    print(f"Downloading {yt.title}...")
    stream.download(output_path=save_path)
    print("Download complete.")



#get inputs
video_url = sys.argv[1]
download_path =  sys.argv[2]

#run
download_video(video_url, download_path)
