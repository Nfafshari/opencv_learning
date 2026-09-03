import cv2
from pathlib import Path

def safe_imread(image_name: str, flag: int = cv2.IMREAD_COLOR) -> cv2.typing.MatLike:
    '''
    safe_imread is a guarded function for cv2.imread so that relative paths do not always return None.

    **IMPORTANT:** Use just the filename (ie. my_image.jpg), this function automatically resolves the path to "/images"
    '''

    # Build the path and verify it exists
    full_path = Path(__file__).resolve().parents[1] / "images" / image_name

    if not full_path.is_file():
        raise FileNotFoundError(f'*ERROR - Could not resolve path {full_path}')

    img = cv2.imread(str(full_path), flag)

    # guard against unknown image paths, which results in img = None
    if img is None:
        raise ValueError(f'*ERROR - Could not read {full_path}')

    return img