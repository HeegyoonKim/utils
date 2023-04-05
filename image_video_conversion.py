import os
import os.path as osp
import cv2
from tqdm import tqdm


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                      '.ppm', '.PPM', '.bmp', '.BMP']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def image2video(input_path, output_path, fps):
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # Set video writer
    img_files = sorted(os.listdir(input_path))
    H, W = cv2.imread(osp.join(input_path, img_files[0])).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W,H))
    
    for img_file in tqdm(img_files, des='Processing image to video'):
        if not is_image_file(img_file):
            continue
        
        img = cv2.imread(osp.join(input_path, img_file))
        writer.write(img)


def video2image(input_path, output_path, ext='jpg'):
    
    os.makedirs(output_path, exist_ok=True)
    
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened() == False:
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for img_id in tqdm(range(n_frames), desc='Processing video to image'):
        is_success, img = cap.read()
        if is_success:
            cv2.imwrite(osp.join(output_path, f'{img_id:06d}.{ext}'), img)
    
    return output_path, fps