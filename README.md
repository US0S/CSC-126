# Auto ID Photo Maker

CSC 126 Computer Vision Midterm Project (Python + OpenCV).
Turns a personal photo into a print-ready ID photo sheet (2x2 inch or passport size).

## CV topics used

| Topic                             | What it does here                                                | Functions                                     |
| --------------------------------- | ---------------------------------------------------------------- | --------------------------------------------- |
| Haar Cascade Classifier           | Detects the face and eyes                                        | `CascadeClassifier`, `detectMultiScale`       |
| Image Analysis and Transformation | Levels the head using the eye line, crops by face ratio, resizes | `getRotationMatrix2D`, `warpAffine`, `resize` |
| Image Segmentation                | Removes the background and replaces it with a solid color        | `grabCut`, morphology, Gaussian blur          |
| Image Operations                  | Lighting fix and tiling on a print sheet                         | `createCLAHE`, array slicing, `rectangle`     |

## Setup

```
pip install -r requirements.txt
```

## Run

Put your own photo in `samples/` (for example `samples/me.jpg`), then:

```
python main.py --input samples/me.jpg
python main.py --input samples/me.jpg --bg blue --size passport --show
python main.py --webcam
python main.py --webcam --camera-flip both
```

Options: `--bg white|blue|red`, `--size 2x2|passport`, `--show`, `--webcam`,
`--camera-flip none|horizontal|vertical|both`.

The webcam capture corrects the common mirrored-camera view by default. Use
`--camera-flip both` if the camera is physically upside down, or
`--camera-flip none` if its orientation is already correct. The live preview
shows face boxes and confirms when a face is detected before capture.

## Output (in `outputs/`)

`01_detection.jpg`, `02_aligned.jpg`, `03_enhanced.jpg`, `04_mask.jpg`,
`05_background_replaced.jpg`, `06_id_photo.jpg`, `07_print_sheet.jpg`

## Tips for best results

Front-facing, evenly lit, plain background, head not tilted too much.

## Author

Your name / group members here
