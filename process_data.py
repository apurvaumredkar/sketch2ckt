import cv2
import os

def preprocess_dir(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(valid_exts):
            continue
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        img = cv2.imread(input_path)
        if img is None:
            print(f"Could not read image: {input_path}")
            continue
        h, w = img.shape[:2]
        new_w = 640
        new_h = int(h * (new_w / w))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(output_path, binary)
        print(f"Processed and saved: {output_path}")

def preprocess_all():
    tasks = [
        ('dataset/train/original', 'dataset/train/processed'),
        ('dataset/test/original', 'dataset/test/processed'),
    ]
    for in_dir, out_dir in tasks:
        if not os.path.exists(in_dir):
            print(f"Input directory does not exist: {in_dir}")
            continue
        print(f"Processing directory: {in_dir} -> {out_dir}")
        preprocess_dir(in_dir, out_dir)

if __name__ == "__main__":
    preprocess_all()
