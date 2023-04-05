import cv2
import numpy as np
from PIL import Image, ImageOps, ImageChops

def add_outline(image_path, outline_width, outline_color, border_width, border_color):
    img = Image.open(image_path).convert("RGBA")

    # Create an RGBA image for the white outline
    white_outline = ImageOps.expand(img, border=outline_width, fill=outline_color)
    img_with_white_outline = Image.new("RGBA", white_outline.size, (0, 0, 0, 0))
    img_with_white_outline.alpha_composite(white_outline)
    img_with_white_outline.alpha_composite(img, (outline_width, outline_width))

    # Create an RGBA image for the black outline
    black_outline = ImageOps.expand(img_with_white_outline, border=border_width, fill=border_color)
    img_with_black_outline = Image.new("RGBA", black_outline.size, (0, 0, 0, 0))
    img_with_black_outline.alpha_composite(black_outline)
    img_with_black_outline.alpha_composite(img_with_white_outline, (border_width, border_width))

    return img_with_black_outline


def apply_perspective_transform(image, output_size, pts):
    h, w = output_size
    dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst_points)
    return cv2.warpPerspective(image, M, (w, h))

def blend_images(background, sticker, x, y, alpha=0.7):
    blended = background.copy()
    h, w, _ = sticker.shape
    sticker_alpha = sticker[:, :, 3] / 255.0

    for c in range(3):
        blended[y:y+h, x:x+w, c] = sticker[:, :, c] * sticker_alpha \
                                 + blended[y:y+h, x:x+w, c] * (1 - sticker_alpha)
    return blended

if __name__ == "__main__":
    input_image_path = "ob.png"
    background_image_path = "wood.jpg"
    output_image_path = "output_image5.png"

    outline_width = 5
    outline_color = (255, 255, 255)  # White
    border_width = 1
    border_color = (0, 0, 0)  # Black

    sticker = add_outline(input_image_path, outline_width, outline_color, border_width, border_color)
    sticker_cv = cv2.cvtColor(np.array(sticker), cv2.COLOR_RGBA2BGRA)

    src_points = np.array([[0, 0], [sticker_cv.shape[1], 0], [sticker_cv.shape[1], sticker_cv.shape[0]], [0, sticker_cv.shape[0]]], dtype=np.float32)
    perspective_points = np.array([[50, 100], [300, 50], [350, 300], [100, 350]], dtype=np.float32)

    transformed_sticker = apply_perspective_transform(sticker_cv, (400, 400), perspective_points)

    background = cv2.imread(background_image_path)
    result = blend_images(background, transformed_sticker, 0, 0)

    cv2.imwrite(output_image_path, result)
