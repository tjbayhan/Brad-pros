# Brad-pros
import os
import requests
from bs4 import BeautifulSoup
import face_recognition
import numpy as np
from urllib.parse import urljoin

# --- CONFIG ---
TARGET_URL = "https://www.eros.com/florida/tampa/files/"
IMAGE_FOLDER = "scraped_images"
KNOWN_FOLDER = "known_faces"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
THRESHOLD = 0.6

# --- FOLDERS ---
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# --- SCRAPE IMAGES FROM EROS ---
def scrape_eros_images(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    image_urls = []

    # Grab images from profile cards
    for img_tag in soup.select("img[src*='/profile_pictures/']"):
        src = img_tag.get("src")
        if src:
            full_url = urljoin(url, src)
            image_urls.append(full_url)

    downloaded = []
    for i, img_url in enumerate(image_urls):
        try:
            img_data = requests.get(img_url, headers=HEADERS).content
            filename = os.path.join(IMAGE_FOLDER, f"eros_tampa_{i}.jpg")
            with open(filename, "wb") as f:
                f.write(img_data)
            downloaded.append(filename)
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")
    return downloaded

# --- LOAD FRIEND IMAGES ---
def load_known_faces(folder):
    known_encodings = []
    known_names = []

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

# --- MATCH AGAINST FRIEND FACES ---
def match_faces(scraped_files, known_encodings, known_names):
    for img_path in scraped_files:
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)

            if distances[best_match_index] < THRESHOLD:
                print(f"[MATCH] {img_path} matches {known_names[best_match_index]}")
            else:
                print(f"[NO MATCH] {img_path}")

# --- RUN EVERYTHING ---
scraped_files = scrape_eros_images(TARGET_URL)
known_enc, known_names = load_known_faces(KNOWN_FOLDER)
match_faces(scraped_files, known_enc, known_names)
