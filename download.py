# -----------------------------------------------------------
# 2.  Download images + metadata via Mapillary API v4
# -----------------------------------------------------------
from pathlib import Path
from tqdm.auto import tqdm
import os, json, requests, time

MAP_TOKEN = os.getenv("MAPILLARY_TOKEN")              # export MAPILLARY_TOKEN="xxxx"
if not MAP_TOKEN:
    raise ValueError("MAPILLARY_TOKEN environment variable not set")

OUT_DIR   = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Test mode: limit to 3 images per cluster for testing
TEST_MODE = True
TEST_IMAGES_PER_CLUSTER = 3

# Top 3 clusters from clustering analysis (output from find_top_clusters.py)
hot_bboxes = [
    {
        "west": -73.928780,
        "south": 40.831735,
        "east": -73.928346,
        "north": 40.832065,
        "cid": 0
    },
    {
        "west": -73.929171,
        "south": 40.831214,
        "east": -73.928737,
        "north": 40.831544,
        "cid": 17
    },
    {
        "west": -73.914938,
        "south": 40.782027,
        "east": -73.914378,
        "north": 40.782447,
        "cid": 3
    }
]

def get_images_in_bbox(bbox, access_token):
    """Get images in bounding box using Mapillary API v4"""
    # API endpoint for images
    url = "https://graph.mapillary.com/images"
    
    # Query parameters
    params = {
        'access_token': access_token,
        'bbox': f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}",
        'limit': 1000  # Maximum limit
    }
    
    headers = {
        'Authorization': f'OAuth {access_token}'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()['data']
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []

def get_image_details(image_id, access_token):
    """Get detailed image information including download URLs"""
    url = f"https://graph.mapillary.com/{image_id}"
    
    params = {
        'access_token': access_token,
        'fields': 'id,captured_at,compass_angle,geometry,thumb_2048_url,thumb_1024_url,thumb_original_url'
    }
    
    headers = {
        'Authorization': f'OAuth {access_token}'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to get details for {image_id}: {e}")
        return None

def save_image(img_id: str, url_2048: str, meta: dict, out_folder: Path):
    """Stream image to disk and write metadata JSON."""
    jpg_path  = out_folder / f"{img_id}.jpg"
    json_path = out_folder / f"{img_id}.json"
    if jpg_path.exists():         # skip if downloaded before
        return
    
    # Add authorization header for image download
    headers = {
        'Authorization': f'OAuth {MAP_TOKEN}'
    }
    
    with requests.get(url_2048, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(jpg_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                f.write(chunk)
    json_path.write_text(json.dumps(meta, indent=2))

for bb in hot_bboxes:
    cid = bb.pop("cid")                           # remove cid key for API call
    print(f"\n⬇︎ Downloading cluster {cid}")
    
    try:
        # Get images using direct API call
        images = get_images_in_bbox(bb, MAP_TOKEN)
        block_dir = OUT_DIR / f"cluster_{cid}"
        block_dir.mkdir(exist_ok=True)

        total_images = len(images)
        print(f"Found {total_images} images in cluster {cid}")
        
        # Limit images for testing
        if TEST_MODE:
            images_to_download = images[:TEST_IMAGES_PER_CLUSTER]
            print(f"TEST MODE: Downloading only {len(images_to_download)} images (limited from {total_images})")
        else:
            images_to_download = images
            print(f"Downloading all {total_images} images")

        for img in tqdm(images_to_download, desc=f"cluster {cid}", unit="img"):
            img_id = img["id"]
            
            # Get detailed image information including download URLs
            print(f"\nGetting details for image {img_id}...")
            image_details = get_image_details(img_id, MAP_TOKEN)
            
            if image_details:
                print(f"Image details: {json.dumps(image_details, indent=2)}")
                
                # Try to get the best available image URL
                image_url = None
                if 'thumb_2048_url' in image_details:
                    image_url = image_details['thumb_2048_url']
                elif 'thumb_1024_url' in image_details:
                    image_url = image_details['thumb_1024_url']
                elif 'thumb_original_url' in image_details:
                    image_url = image_details['thumb_original_url']
                
                if image_url:
                    meta = {
                        "id": img_id,
                        "capturedAt": image_details.get("captured_at"),
                        "compass": image_details.get("compass_angle"),
                        "lat": image_details.get("geometry", {}).get("coordinates", [0, 0])[1],
                        "lon": image_details.get("geometry", {}).get("coordinates", [0, 0])[0],
                        "cluster": cid,
                        "image_url": image_url,
                        "full_response": image_details
                    }
                    
                    try:
                        save_image(img_id, image_url, meta, block_dir)
                        print(f" ✓ Downloaded {img_id}")
                    except Exception as e:
                        print(f" ! Error downloading {img_id}: {e}")
                else:
                    print(f" ! No image URL found for {img_id}")
            else:
                print(f" ! Failed to get details for {img_id}")
            
            time.sleep(0.5)  # Rate limiting
                
    except Exception as e:
        print(f"Error processing cluster {cid}: {e}")
        continue

print("✅  Download complete")
if TEST_MODE:
    print(f"🧪 TEST MODE: Limited to {TEST_IMAGES_PER_CLUSTER} images per cluster")
    print("To download all images, set TEST_MODE = False")
