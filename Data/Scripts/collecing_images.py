import yt_dlp
import cv2
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_video(url: str, output_dir: str, num_frames: int, sharpness_threshold: float):
    """
    Télécharge une vidéo et extrait les frames (fonction worker pour parallélisation).
    """
    try:
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'extractor_args': {'youtube': {'player_client': ['android']}},
           # 'cookiesfrombrowser': ('firefox',)
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            video_id = info.get('id', 'unknown')
        
        cap = cv2.VideoCapture(filename)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                    frame_filename = os.path.join(output_dir, f"{video_id}_frame_{saved_frames_count}.png")
                    cv2.imwrite(frame_filename, frame)
            
            attempts += 1
    
        cap.release()
        os.remove(filename)
        
        return f"Terminé: {video_id}"
    
    except Exception as e:
        return f"Erreur pour {url}: {str(e)}"


def download_and_extract_frames(urls: list, output_dir: str, num_frames: int, sharpness_threshold: float = 150.0, max_workers: int = 20):
    """
    Download YouTube videos and extract random frames in parallel.
    """
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_video, url, output_dir, num_frames, sharpness_threshold): url 
            for url in urls
        }
        
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Erreur pour {url}: {str(e)}")


if __name__ == "__main__":

    with open('Data/train_urls.txt', 'r') as f:
        train_urls = [line.strip() for line in f.readlines()]

    download_and_extract_frames(urls=train_urls,
                                output_dir="Data/train_images",
                                num_frames=5,
                                sharpness_threshold=150,
                                max_workers=32)
    
    with open('Data/test_urls.txt', 'r') as f:
        test_urls = [line.strip() for line in f.readlines()]

    download_and_extract_frames(urls=test_urls,
                                output_dir="Data/test_qualite",
                                num_frames=5,
                                sharpness_threshold=150,
                                max_workers=32)