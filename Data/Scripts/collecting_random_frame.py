import yt_dlp
import cv2
import random
import os
import numpy as np

def download_youtube_videos(urls: list, output_dir: str = "videos") -> list:
    """
    Download multiple YouTube videos given a list of URLs.
    Returns a list of downloaded video file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    downloaded_paths = []
    for url in urls:
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'format': 'bestvideo+bestaudio/best',
            'quiet': True,
            'merge_output_format': 'mp4',
            'cookiesfrombrowser': ['firefox'],
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            },
         }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if not filename.endswith(".mp4"):
                filename = os.path.splitext(filename)[0] + ".mp4"
            downloaded_paths.append(filename)

    return downloaded_paths


def extract_random_frames(video_paths: list, output_dir: str, num_frames: int, sharpness_threshold: float = 100.0):
    """
    Extract `num_frames` random frames from each video in `video_paths`
    and save them directly into `output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        saved_frames_count = 0
        max_attempts = num_frames * 5 
        attempts = 0

        while saved_frames_count < num_frames and attempts < max_attempts:
            frame_id = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success, frame = cap.read()

            if success:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
                
                if laplacian_var > sharpness_threshold:
                    saved_frames_count += 1
                    filename = os.path.join(output_dir, f"{video_name}_frame_{saved_frames_count}.jpg")
                    cv2.imwrite(filename, frame)
            
            attempts += 1
    
        cap.release()


if __name__ == "__main__":
    
    # Official highligts of the seasons 2021 to 2023 that last arround 40 minutes each
    taining_urls = [
        "https://www.youtube.com/watch?v=-zcIsk6GE6Q",
        "https://www.youtube.com/watch?v=dxbVdyVYGcQ",
        "https://www.youtube.com/watch?v=hFrIVlkTDMs",
    ]

     # Official highligts of the seasons 2024
    test_url = ["https://www.youtube.com/watch?v=_e2tSzsuark"]

    train_videos_path = "Data/train_videos"
    train_file_paths = [os.path.join(train_videos_path, f) for f in os.listdir(train_videos_path) if os.path.isfile(os.path.join(train_videos_path, f))]
    
    extract_random_frames(video_paths=train_file_paths, 
                          output_dir="Data/train_images", 
                          num_frames=500,
                          sharpness_threshold=200)