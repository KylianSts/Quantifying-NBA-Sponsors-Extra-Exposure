import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_video_info(video_id: str, start_date: str, end_date: str, min_duration_secs: int):
    """
    Extrait les infos d'une vidéo et vérifie si elle correspond aux critères.
    """
    try:
        ydl_opts = {
            'quiet': True,
            'ignoreerrors': True,
        }
        
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            if info:
                upload_date = info.get('upload_date', '')
                duration = info.get('duration', 0)
                
                if upload_date and start_date <= upload_date <= end_date:
                    if duration and duration > min_duration_secs:
                        return video_url
        
        return None
    
    except Exception as e:
        print(f"Erreur pour {video_id}: {str(e)}")
        return None


def get_video_urls(channel_url, outputdir, start_date, end_date, min_duration_secs, max_workers: int = 20):
    """
    Récupère les URLs des vidéos d'une chaîne qui correspondent à des filtres de date et de durée.
    Version parallélisée.
    """
    
    ydl_opts = {
        'quiet': False,          
        'ignoreerrors': True,  
        'extract_flat': True,  # Juste pour avoir les IDs
    }

    video_ids = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(channel_url, download=False)
        if 'entries' in info_dict:
            for video_entry in info_dict['entries']:
                if video_entry and 'id' in video_entry:
                    video_ids.append(video_entry['id'])
    
    print(f"Total de vidéos trouvées: {len(video_ids)}")
    print("Filtrage en parallèle...")
    
    video_urls = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_video_info, video_id, start_date, end_date, min_duration_secs): video_id 
            for video_id in video_ids
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                video_urls.append(result)
                print(f"Vidéo valide trouvée: {len(video_urls)}")

    with open(outputdir, 'w') as f:
        for url in video_urls: 
            f.write(url + '\n')

    return video_urls 


if __name__ == "__main__":
    
    CHANNEL_URL = "https://www.youtube.com/@TheGametimeHighlights/videos"
    MINIMUM_DURATION = 8 * 60
    START_DATE_TRAIN = "20241101" 
    END_DATE_TRAIN = "20250501"  
    START_DATE_TEST = "20251020" 
    END_DATE_TEST = "20251201"  
    
    train_urls = get_video_urls(
        channel_url=CHANNEL_URL,
        outputdir='Data/train_urls.txt',
        start_date=START_DATE_TRAIN,
        end_date=END_DATE_TRAIN,
        min_duration_secs=MINIMUM_DURATION,
        max_workers=32
    )
    
    print(f"Total: {len(train_urls)} URLs sauvegardées")

    test_urls = get_video_urls(
        channel_url=CHANNEL_URL,
        outputdir='Data/test_urls.txt',
        start_date=START_DATE_TEST,
        end_date=END_DATE_TEST,
        min_duration_secs=MINIMUM_DURATION,
        max_workers=32
    )
    
    print(f"Total: {len(test_urls)} URLs sauvegardées")