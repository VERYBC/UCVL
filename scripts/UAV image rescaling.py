import os
import shutil
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

root_dir = '../data/Real-world filght data'

object = 'drone'

DATA_LIST = [3]

target_size = (392, 392)

img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def process_image(args):
    img_path, save_path = args
    if os.path.exists(save_path):
        return None
    try:
        with Image.open(img_path) as img:
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(save_path)
    except Exception as e:
        return f"Error processing {img_path}: {e}"
    return None


for i in range(1, 30):
    if i not in DATA_LIST:
        continue

    folder_name = f"{i:02d}"   # 01, 02, ...
    drone_dir = os.path.join(root_dir, folder_name, object)
    output_dir = os.path.join(root_dir, folder_name, object + "_392")
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(drone_dir)
    img_files = [f for f in files if f.lower().endswith(img_exts)]
    other_files = [f for f in files if not f.lower().endswith(img_exts)]

    for f in other_files:
        src_path = os.path.join(drone_dir, f)
        dst_path = os.path.join(output_dir, f)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    tasks = [(os.path.join(drone_dir, f), os.path.join(output_dir, f)) for f in img_files]

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_image, tasks),
                            total=len(tasks),
                            desc=f"Processing {folder_name}",
                            unit="img"))



