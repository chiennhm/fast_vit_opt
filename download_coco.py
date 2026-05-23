#
# MS COCO 2017 Dataset Downloader and Extractor
#
# Downloads and extracts:
#   1. Train Images (19 GB)
#   2. Val Images (1 GB)
#   3. Train/Val Annotations (241 MB)
#   4. Test Images (6.3 GB) [Optional]
#   5. Test Image Info (1 MB) [Optional]
#
# Structure created:
#   ./data/coco/
#     ├── train2017/
#     ├── val2017/
#     ├── test2017/
#     └── annotations/
#

import os
import sys
import time
import zipfile
import urllib.request

# URLs for MS COCO 2017
COCO_URLS = {
    "annotations": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "check_path": "annotations/instances_train2017.json",
        "descr": "Annotations (Train/Val)"
    },
    "val2017": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "check_path": "val2017",
        "descr": "Validation Images (~1 GB)"
    },
    "train2017": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "check_path": "train2017",
        "descr": "Training Images (~19 GB)"
    },
    "test_info": {
        "url": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
        "check_path": "annotations/image_info_test2017.json",
        "descr": "Test Info Annotations (~1 MB)"
    },
    "test2017": {
        "url": "http://images.cocodataset.org/zips/test2017.zip",
        "check_path": "test2017",
        "descr": "Test Images (~6.3 GB)"
    }
}

def format_size(bytes_size):
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def download_with_progress(url, dest_path, descr):
    """Download a file showing progress bar and download speed."""
    print(f"\nDownloading {descr} from:\n  {url}")
    start_time = time.time()
    
    # Custom opener to set User-Agent (sometimes needed to avoid HTTP 403)
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    
    try:
        response = urllib.request.urlopen(url)
        meta = response.info()
        file_size = int(meta.get("Content-Length", 0))
        print(f"Total size: {format_size(file_size)}")
        
        downloaded = 0
        block_size = 1024 * 1024  # 1 MB blocks
        
        with open(dest_path, "wb") as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                
                downloaded += len(buffer)
                f.write(buffer)
                
                # Calculate speed and ETA
                elapsed = time.time() - start_time
                speed = downloaded / elapsed if elapsed > 0 else 0
                eta = (file_size - downloaded) / speed if speed > 0 else 0
                
                # Format progress string
                percent = (downloaded / file_size) * 100 if file_size > 0 else 0
                bar_len = 30
                filled_len = int(round(bar_len * downloaded / float(file_size))) if file_size > 0 else 0
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                
                status = f"\rProgress: |{bar}| {percent:5.1f}% ({format_size(downloaded)}/{format_size(file_size)}) | {format_size(speed)}/s | ETA: {eta:.0f}s"
                sys.stdout.write(status)
                sys.stdout.flush()
                
        print(f"\nDownload complete in {time.time() - start_time:.1f}s.")
        return True
    except Exception as e:
        print(f"\nError downloading {descr}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the target directory."""
    print(f"Extracting {os.path.basename(zip_path)} to {extract_to}...")
    start_time = time.time()
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files to show extraction progress
            members = zip_ref.infolist()
            total_files = len(members)
            
            for i, member in enumerate(members):
                zip_ref.extract(member, extract_to)
                if i % max(1, total_files // 100) == 0 or i == total_files - 1:
                    percent = (i + 1) / total_files * 100
                    sys.stdout.write(f"\rExtraction progress: {percent:.1f}% ({i + 1}/{total_files} files)")
                    sys.stdout.flush()
                    
        print(f"\nExtraction complete in {time.time() - start_time:.1f}s.")
        return True
    except Exception as e:
        print(f"\nError extracting {zip_path}: {e}")
        return False

def main():
    # Setup directories
    dest_dir = "./data/coco"
    os.makedirs(dest_dir, exist_ok=True)
    
    # Parse arguments
    args = [arg.lower() for arg in sys.argv[1:]]
    
    download_keys = ["annotations", "val2017", "train2017"]  # Default
    mode_descr = "Train + Val + Annotations"
    
    if "--only-val" in args:
        download_keys = ["annotations", "val2017"]
        mode_descr = "Val + Annotations Only (~1.2 GB)"
    elif "--only-test" in args:
        download_keys = ["test_info", "test2017"]
        mode_descr = "Test + Test Info Only (~6.3 GB)"
    elif "--include-test" in args or "--test" in args:
        download_keys = ["annotations", "val2017", "train2017", "test_info", "test2017"]
        mode_descr = "Train + Val + Test + All Annotations (~27.5 GB)"
        
    print("============================================================")
    print("  Tải và giải nén Bộ dữ liệu MS COCO 2017")
    print("============================================================")
    print(f"Thư mục đích: {os.path.abspath(dest_dir)}")
    print(f"Chế độ tải:  {mode_descr}")
    print("------------------------------------------------------------")
    
    for key in download_keys:
        info = COCO_URLS[key]
        expected_path = os.path.join(dest_dir, info["check_path"])
        
        if os.path.exists(expected_path):
            print(f"[OK] Thư mục/Tệp '{info['check_path']}' đã tồn tại. Bỏ qua.")
            continue
            
        zip_name = os.path.basename(info["url"])
        zip_path = os.path.join(dest_dir, zip_name)
        
        # Download
        success = download_with_progress(info["url"], zip_path, info["descr"])
        if not success:
            print(f"Thất bại khi tải {info['descr']}. Dừng tiến trình.")
            sys.exit(1)
            
        # Extract
        success = extract_zip(zip_path, dest_dir)
        if not success:
            print(f"Thất bại khi giải nén {zip_name}. Dừng tiến trình.")
            sys.exit(1)
            
        # Cleanup zip file
        print(f"Đang xóa file zip tạm: {zip_name}...")
        try:
            os.remove(zip_path)
            print("Xóa thành công.")
        except Exception as e:
            print(f"Không thể xóa file zip: {e}")
            
    print("\n============================================================")
    print("  Hoàn thành! Các phần được chọn của MS COCO 2017 đã sẵn sàng.")
    print("============================================================")

if __name__ == "__main__":
    main()
