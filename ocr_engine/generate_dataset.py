import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil

OUTPUT_DIR = "templates"
FONTS_DIR = "fonts"
IMG_HEIGHT = 64 

FONT_NIK = os.path.join(FONTS_DIR, "OCRAEXT.TTF")

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    try:
        font = ImageFont.truetype(FONT_NIK, 48)
    except IOError:
        return

    for i in range(10):
        char = str(i)
        
        image = Image.new('L', (IMG_HEIGHT, IMG_HEIGHT), 0)
        draw = ImageDraw.Draw(image)

        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        x = (IMG_HEIGHT - w) / 2
        y = (IMG_HEIGHT - h) / 2
        
        draw.text((x, y), char, font=font, fill=255)

        img_np = np.array(image)
        coords = cv2.findNonZero(img_np)
        x, y, w, h = cv2.boundingRect(coords)
        crop = img_np[y:y+h, x:x+w]
        
        cv2.imwrite(f"{OUTPUT_DIR}/{char}.jpg", crop)

if __name__ == "__main__":
    main()