import yt_dlp
import cv2
import random
import os
import yt_dlp
import os

def get_video_urls(channel_url, start_date, end_date, min_duration_secs):
    """
    Récupère les URLs des vidéos d'une chaîne qui correspondent à des filtres de date et de durée.
    """
    
    ydl_opts = {
        'quiet': True,          
        'ignoreerrors': True,  
        'extract_flat': False,  
    }

    video_urls = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            
        info_dict = ydl.extract_info(channel_url, download=False)
        if 'entries' in info_dict:
             for video_entry in info_dict['entries']:
                 if video_entry:
                     upload_date = video_entry.get('upload_date', '')
                     if upload_date and start_date <= upload_date <= end_date:
                         duration = video_entry.get('duration', 0)
                         if duration and duration > min_duration_secs:
                             video_urls.append(f"https://www.youtube.com/watch?v={video_entry['id']}")

    return video_urls

def download_and_extract_frames(urls: list, output_dir: str, num_frames: int, sharpness_threshold: float = 150.0):
    """
    Download YouTube videos and extract random frames on-the-fly without storing all videos.
    """
    os.makedirs(output_dir, exist_ok=True)

    for url in urls:
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'format': 'best[ext=mp4]/best',
            'extractor_args': {'youtube': {'player_client': ['android']}},
           # 'cookiesfrombrowser': ('firefox',)
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        
        cap = cv2.VideoCapture(filename)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.splitext(os.path.basename(filename))[0]

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
                    frame_filename = os.path.join(output_dir, f"{video_name}_frame_{saved_frames_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
            
            attempts += 1
    
        cap.release()
        
        os.remove(filename)

if __name__ == "__main__":
    
    # Set of unofficial highlights of the 2024/2025 season
    CHANNEL_URL = "https://www.youtube.com/@TheGametimeHighlights/videos"
    MINIMUM_DURATION = 8 * 60
    START_DATE = "20241101" 
    END_DATE = "20250501"  
    
    train_urls = get_video_urls(
        channel_url=CHANNEL_URL,
        start_date=START_DATE,
        end_date=END_DATE,
        min_duration_secs=MINIMUM_DURATION
    )

    print(len(train_urls))

    download_and_extract_frames(urls=train_urls,
                                output_dir="Data/train_images",
                                num_frames=10,
                                sharpness_threshold=150)

    """# Set of unofficial highlights of the 2025/2026 season
    test_url = []

    train_videos_path = "Data/train_videos"
    train_file_paths = [os.path.join(train_videos_path, f) for f in os.listdir(train_videos_path) if os.path.isfile(os.path.join(train_videos_path, f))]
    
    train_paths = download_youtube_videos(train_urls, "Data/train_videos")

    extract_random_frames(video_paths=train_file_paths, 
                          output_dir="Data/train_images", 
                          num_frames=50,
                          sharpness_threshold=150)
    
    test_paths = download_youtube_videos(test_url, "Data/test_videos")

    extract_random_frames(video_paths=test_paths, 
                          output_dir="Data/test_images", 
                          num_frames=50,
                          sharpness_threshold=150)"""
