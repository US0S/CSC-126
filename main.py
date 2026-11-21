"""
Auto ID Photo Maker  -  CSC 126 Computer Vision Midterm
-------------------------------------------------------
Takes a personal photo and produces a print-ready ID photo sheet.

Pipeline (CV topics used):
  1. Image Operations ............ load, resize, CLAHE brightness/contrast, tiling
  2. Haar Cascade Classifier ..... detect face + eyes
  3. Analysis & Transformation ... rotate to level the eyes, crop by face ratio, resize
  4. Image Segmentation .......... GrabCut background removal -> solid color

Usage:
  python main.py --input samples/me.jpg
  python main.py --input samples/me.jpg --bg blue --size passport --show
  python main.py --webcam            # press SPACE to capture, ESC to quit
"""

import argparse
import os
import sys

import cv2
import numpy as np

OUT_DIR = "outputs"

BG_COLORS = {  # BGR
    "white": (255, 255, 255),
    "blue": (204, 102, 0),
    "red": (0, 0, 200),
}

# name: (width_ratio, height_ratio, output_px_w, output_px_h)  @ 300 DPI
PHOTO_SIZES = {
    "2x2": (600, 600),        # 2 x 2 inch
    "passport": (413, 531),   # 35 x 45 mm
}


# ---------------------------------------------------------------- 1. IMAGE OPERATIONS
def load_image(path, max_side=900):
    """Read the image and shrink it so processing stays fast."""
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"[error] Could not read image: {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def enhance(img):
    """Fix lighting with CLAHE on the L channel (keeps colors natural)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def make_sheet(photo, sheet_w=1800, sheet_h=1200, gap=20, margin=30):
    """Tile the ID photo on a 6x4 inch (300 DPI) sheet, with thin cut lines."""
    ph, pw = photo.shape[:2]
    cols = (sheet_w - 2 * margin + gap) // (pw + gap)
    rows = (sheet_h - 2 * margin + gap) // (ph + gap)
    sheet = np.full((sheet_h, sheet_w, 3), 255, np.uint8)

    grid_w = cols * pw + (cols - 1) * gap
    grid_h = rows * ph + (rows - 1) * gap
    x0 = (sheet_w - grid_w) // 2
    y0 = (sheet_h - grid_h) // 2

    for r in range(rows):
        for c in range(cols):
            x = x0 + c * (pw + gap)
            y = y0 + r * (ph + gap)
            sheet[y:y + ph, x:x + pw] = photo
            cv2.rectangle(sheet, (x, y), (x + pw - 1, y + ph - 1), (180, 180, 180), 1)
    return sheet, rows * cols


# ---------------------------------------------------------------- 2. HAAR CASCADE
def detect_face_and_eyes(img):
    """Return (face_box, [eye_boxes]) using Haar cascades. Boxes are (x, y, w, h)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        sys.exit("[error] OpenCV face cascade could not be loaded.")
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml")
    if eye_cascade.empty():
        sys.exit("[error] OpenCV eye cascade could not be loaded.")

    faces = []
    for source in (gray, equalized):
        for scale_factor, neighbors in ((1.05, 6), (1.1, 5)):
            detected = face_cascade.detectMultiScale(
                source, scaleFactor=scale_factor, minNeighbors=neighbors,
                minSize=(80, 80))
            faces.extend(tuple(face) for face in detected)
    if len(faces) == 0:
        return None, []

    # Prefer the largest face, which is normally the subject in an ID photo.
    face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = face

    # Eyes are only in the upper part of the face -> smaller search area, fewer false hits
    roi_y1 = fy
    roi_y2 = min(gray.shape[0], fy + int(fh * 0.6))
    roi_x1 = max(0, fx)
    roi_x2 = min(gray.shape[1], fx + fw)
    eye_candidates = []
    for source in (gray, equalized):
        roi = source[roi_y1:roi_y2, roi_x1:roi_x2]
        detected = eye_cascade.detectMultiScale(
            roi, scaleFactor=1.05, minNeighbors=7,
            minSize=(max(fw // 12, 8), max(fw // 12, 8)))
        eye_candidates.extend(
            (roi_x1 + ex, roi_y1 + ey, ew, eh)
            for ex, ey, ew, eh in detected)

    # Keep plausible, non-overlapping candidates and choose the pair with the
    # strongest size and left/right separation.
    candidates = []
    for eye in eye_candidates:
        ex, ey, ew, eh = eye
        if ey + eh / 2 <= fy + fh * 0.55 and fw * 0.06 <= ew <= fw * 0.45:
            if not any(
                abs(ex - other[0]) < 0.25 * ew
                and abs(ey - other[1]) < 0.25 * eh
                for other in candidates
            ):
                candidates.append(eye)

    best_pair = None
    best_score = float("-inf")
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            left, right = sorted((first, second), key=lambda eye: eye[0])
            left_center = (left[0] + left[2] / 2, left[1] + left[3] / 2)
            right_center = (right[0] + right[2] / 2, right[1] + right[3] / 2)
            separation = right_center[0] - left_center[0]
            vertical_gap = abs(right_center[1] - left_center[1])
            if separation < 0.2 * fw or vertical_gap > 0.25 * fh:
                continue
            score = (
                min(left[2] * left[3], right[2] * right[3])
                + separation * 0.2
                - vertical_gap * 0.5
            )
            if score > best_score:
                best_pair, best_score = [left, right], score
    eyes = best_pair or []
    return tuple(face), eyes


def draw_detections(img, face, eyes):
    vis = img.copy()
    x, y, w, h = face
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for ex, ey, ew, eh in eyes:
        cv2.rectangle(vis, (ex, ey), (ex + ew, ey + eh), (255, 128, 0), 2)
    return vis


# ---------------------------------------------------------------- 3. TRANSFORMATION
def align_by_eyes(img, face, eyes):
    """Rotate the image so the line between the eyes is horizontal."""
    if len(eyes) < 2:
        print("[warn] Could not find both eyes - skipping alignment.")
        return img, face, 0.0

    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
    c1 = (x1 + w1 / 2, y1 + h1 / 2)
    c2 = (x2 + w2 / 2, y2 + h2 / 2)
    angle = float(np.degrees(np.arctan2(c2[1] - c1[1], c2[0] - c1[0])))

    center = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    h, w = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # Move the face box with the same rotation using all corners.
    fx, fy, fw, fh = face
    corners = np.array([
        [fx, fy, 1.0],
        [fx + fw, fy, 1.0],
        [fx, fy + fh, 1.0],
        [fx + fw, fy + fh, 1.0],
    ])
    rotated_corners = corners @ M.T
    min_x, min_y = np.min(rotated_corners, axis=0)
    max_x, max_y = np.max(rotated_corners, axis=0)
    new_face = (
        max(0, int(min_x)),
        max(0, int(min_y)),
        min(w - max(0, int(min_x)), max(1, int(max_x - min_x))),
        min(h - max(0, int(min_y)), max(1, int(max_y - min_y))),
    )
    return rotated, new_face, angle


def crop_to_id(img, face, out_size, bg_color):
    """Crop around the face so the head fills ~half the photo height, then resize."""
    out_w, out_h = out_size
    ratio = out_w / out_h
    fx, fy, fw, fh = face

    crop_h = fh * 2.2
    crop_w = crop_h * ratio
    cx = fx + fw / 2
    cy = fy + fh * 0.55

    x0, x1 = int(cx - crop_w / 2), int(cx + crop_w / 2)
    y0, y1 = int(cy - crop_h / 2), int(cy + crop_h / 2)

    # Pad with the background color if the crop goes outside the image
    h, w = img.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    if any((pad_l, pad_t, pad_r, pad_b)):
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                 cv2.BORDER_CONSTANT, value=bg_color)
        x0, x1, y0, y1 = x0 + pad_l, x1 + pad_l, y0 + pad_t, y1 + pad_t

    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 4. SEGMENTATION
def replace_background(img, face, bg_color):
    """GrabCut segmentation seeded from the detected face, then paste on a solid color."""
    h, w = img.shape[:2]
    fx, fy, fw, fh = face

    def clamp(v, lo, hi):
        return int(max(lo, min(hi, v)))

    mask = np.full((h, w), cv2.GC_BGD, np.uint8)

    # Probably foreground: head + shoulders region around the face
    x0, x1 = clamp(fx - 0.8 * fw, 0, w), clamp(fx + 1.8 * fw, 0, w)
    y0 = clamp(fy - 0.5 * fh, 0, h)
    mask[y0:h, x0:x1] = cv2.GC_PR_FGD
    
    # Always keep a thin border as sure background (close-up photos can fill the frame)
    bw = max(int(0.04 * w), 5)
    bh = max(int(0.04 * h), 5)
    mask[:bh, :] = cv2.GC_BGD
    mask[:, :bw] = cv2.GC_BGD
    mask[:, w - bw:] = cv2.GC_BGD

    # Definitely foreground: the middle of the face
    mask[clamp(fy + 0.2 * fh, 0, h):clamp(fy + 0.9 * fh, 0, h),
         clamp(fx + 0.2 * fw, 0, w):clamp(fx + 0.8 * fw, 0, w)] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Clean up the mask, then soften the edge
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.GaussianBlur(fg, (7, 7), 0).astype(np.float32) / 255.0
    alpha = alpha[..., None]

    bg = np.full_like(img, bg_color)
    out = (img * alpha + bg * (1 - alpha)).astype(np.uint8)
    return out, fg


# ---------------------------------------------------------------- INPUT HELPERS
def apply_camera_flip(frame, mode):
    """Correct common webcam orientation problems before preview and saving."""
    if mode == "horizontal":
        return cv2.flip(frame, 1)
    if mode == "vertical":
        return cv2.flip(frame, 0)
    if mode == "both":
        return cv2.flip(frame, -1)
    return frame


def capture_from_webcam(save_path="samples/webcam_capture.jpg", flip_mode="horizontal"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[error] Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        cap.release()
        sys.exit("[error] OpenCV face cascade could not be loaded.")
    print("Webcam open: press SPACE to capture, ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("[error] Could not read a frame from the webcam.")
        frame = apply_camera_flip(frame, flip_mode)
        preview = frame.copy()
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))
        for x, y, w, h in faces:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(preview, "SPACE = capture | ESC = quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        face_detected = len(faces) > 0
        status = "Face detected - ready" if face_detected else "Move into the guide"
        cv2.putText(preview, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255) if face_detected else (0, 165, 255), 2)
        cv2.imshow("Webcam", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if not cv2.imwrite(save_path, frame):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(f"[error] Could not save webcam image: {save_path}")
            break
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    cap.release()
    cv2.destroyAllWindows()
    return save_path


def save(name, img):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser(description="Auto ID Photo Maker")
    ap.add_argument("--input", help="path to your photo")
    ap.add_argument("--webcam", action="store_true", help="capture the photo from webcam")
    ap.add_argument(
        "--camera-flip",
        choices=("none", "horizontal", "vertical", "both"),
        default="horizontal",
        help="correct webcam orientation (default: horizontal mirror correction)",
    )
    ap.add_argument("--bg", choices=BG_COLORS.keys(), default="white")
    ap.add_argument("--size", choices=PHOTO_SIZES.keys(), default="2x2")
    ap.add_argument("--show", action="store_true", help="display each step in a window")
    args = ap.parse_args()

    if args.webcam:
        args.input = capture_from_webcam(flip_mode=args.camera_flip)
    if not args.input:
        ap.error("provide --input <photo> or use --webcam")

    bg_color = BG_COLORS[args.bg]

    # 1. Load + detect (Haar)
    img = load_image(args.input)
    face, eyes = detect_face_and_eyes(img)
    if face is None:
        sys.exit("[error] No face found. Use a front-facing, well-lit photo.")
    print(f"[info] Face found. Eyes detected: {len(eyes)}")
    detection = draw_detections(img, face, eyes)
    save("01_detection.jpg", detection)

    # 2. Align (transformation)
    aligned, face, angle = align_by_eyes(img, face, eyes)
    print(f"[info] Rotated by {angle:.2f} degrees")
    save("02_aligned.jpg", aligned)

    # 3. Brightness / contrast (image operations)
    enhanced = enhance(aligned)
    save("03_enhanced.jpg", enhanced)

    # 4. Background removal (segmentation)
    segmented, mask = replace_background(enhanced, face, bg_color)
    save("04_mask.jpg", mask)
    save("05_background_replaced.jpg", segmented)

    # 5. Crop + resize, then tile on a print sheet
    photo = crop_to_id(segmented, face, PHOTO_SIZES[args.size], bg_color)
    save("06_id_photo.jpg", photo)
    sheet, count = make_sheet(photo)
    save("07_print_sheet.jpg", sheet)
    print(f"[done] Saved {count} photos on a 6x4 inch sheet in '{OUT_DIR}/'")

    if args.show:
        for title, im in [("1 Detection", detection), ("2 Mask", mask),
                          ("3 ID Photo", photo), ("4 Sheet", sheet)]:
            cv2.imshow(title, im)
        print("Press any key in a window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
