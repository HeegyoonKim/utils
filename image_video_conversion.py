import os
import os.path as osp
import cv2
from tqdm import tqdm


def is_image_file(
    filename: str
) -> bool:
    """
    Check the input is image file

    Args:
        filename (str): path to file

    Returns:
        bool: True or False
    """
    
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                      '.ppm', '.PPM', '.bmp', '.BMP']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def image2video(
    input_path: str, output_path: str, fps: int
) -> None:
    """
    Create video from a sequence of images

    Args:
        input_path (str): path to image directory
        output_path (str): output path with video extension
        fps (int): frames per second
    """
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # Set video writer
    img_files = sorted(os.listdir(input_path))
    H, W = cv2.imread(osp.join(input_path, img_files[0])).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W,H))
    
    for img_file in tqdm(img_files):
        if not is_image_file(img_file):
            continue
        
        img = cv2.imread(osp.join(input_path, img_file))
        writer.write(img)


def video2image(
    input_path: str, output_path: str, ext: str = 'jpg'
) -> None:
    """
    Extract images from video

    Args:
        input_path (str): path to video
        output_path (str): path to image directory
        ext (str, optional): image extension. Defaults to 'jpg'.
    """
    
    os.makedirs(output_path, exist_ok=True)
    
    # Set video reader
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened() == False:
        return None
    
    for img_id in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        is_success, img = cap.read()
        if is_success:
            cv2.imwrite(osp.join(output_path, f'{img_id:06d}.{ext}'), img)