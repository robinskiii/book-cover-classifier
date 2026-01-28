import requests
import os
import time

#create dataset folder
FOLDER = 'dataset'

#sample size
IMAGES_PER_GENRE = 150

GENRES = {
    "science_fiction": "Science_Fiction", "fantasy": "Fantasy",
    "romance": "Romance", "cookbooks": "Cookbooks", "architecture": "Architecture"
}

def get_books_by_subject(subject_key, target_count):
    """
    Open Library Subject API. 
    We ask for double the target count because many entries lack cover IDs.
    """
    limit = target_count * 2
    url = f"https://openlibrary.org/subjects/{subject_key}.json?limit={limit}"
    try:
        print(f"   Requesting list from: {url}")
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return data.get('works', [])
    except Exception as e:
        print(f"   Error fetching subject list: {e}")
    return []

def download_cover(cover_id, save_path):
    """
    Downloads the Large (L) cover image using its ID.
    Returns True if successful and the image is valid.
    """
    url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    try:
        #5 second timeout for downloading the image
        r = requests.get(url, timeout=5)
        #check status and ensure it's bigger than 1000 bytes
        if r.status_code == 200 and len(r.content) > 1000:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

def main():
    print("--- Starting Direct Open Library Download ---")
    print(f"Target: {IMAGES_PER_GENRE} images for {len(GENRES)} genres.")
    
    os.makedirs(FOLDER, exist_ok=True)

    for subject_key, folder_name in GENRES.items():
        print(f"\n--- Processing Genre: {folder_name} (Subject: {subject_key}) ---")
        
        #create genre folder
        genre_path = os.path.join(FOLDER, folder_name)
        os.makedirs(genre_path, exist_ok=True)
        
        #get list of books
        books = get_books_by_subject(subject_key, IMAGES_PER_GENRE)
        print(f"   API returned {len(books)} candidates. Starting download process...")
        count = 0

        #check existing files to avoid over-download
        existing_files = len([name for name in os.listdir(genre_path) if os.path.isfile(os.path.join(genre_path, name))])
        count = existing_files
        print(f"   Found {existing_files} existing images. Need {IMAGES_PER_GENRE - count} more.")

        for book in books:
            #stop once we reach the image count
            if count >= IMAGES_PER_GENRE:
                break
                
            title = book.get('title', 'Unknown Title')
            cover_id = book.get('cover_id')
            
            #skip if entry has no cover image available
            if not cover_id:
                continue
                
            save_filename = os.path.join(genre_path, f"{cover_id}.jpg")
            
            #skip if we already downloaded this specific image
            if os.path.exists(save_filename):
                continue
            
            #print download progress
            print(f"   [{count+1}/{IMAGES_PER_GENRE}] Downloading: {title[:50]:<50}", end="\r")
            
            success = download_cover(cover_id, save_filename)
            
            if success:
                count += 1
                time.sleep(0.1)
                
        print(f"\n   Finished {folder_name}. Total available: {count}")
        time.sleep(1)

    print("\n--- dowload complete ---")

if __name__ == "__main__":
    main()