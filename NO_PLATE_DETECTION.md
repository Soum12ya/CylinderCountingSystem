# Truck Number Plate Detection and Recognition

## 1. Overview

The Truck Number Plate Detection System identifies incoming trucks at the plant entry gate using an industrial camera and Artificial Intelligence.

When a truck approaches the entry gate, the camera captures its registration plate. The AI system detects the plate region, extracts the plate image, and uses Optical Character Recognition (OCR) to convert the registration characters into text.

Django receives the recognized registration number, searches the vehicle database, and allows the gate operator to verify or register the vehicle.

**Main objective:** Automatically identify incoming trucks, reduce manual data entry, and create accurate vehicle-entry records.

### Proposed Technology

| Component | Technology |
|---|---|
| Backend | Python and Django |
| Database | PostgreSQL |
| Camera Integration | OpenCV and RTSP |
| Number Plate Detection | YOLO |
| Character Recognition | PaddleOCR |
| Image Processing | OpenCV |
| Model Training | PyTorch |
| Frontend | Django Templates, HTML, CSS, JavaScript |

---

# 2. Overall System Workflow

```text
                  TRUCK ARRIVES
                        |
                        v
                  ENTRY CAMERA
                        |
                        v
                 IMAGE CAPTURE
                        |
                        v
              IMAGE QUALITY CHECK
                        |
                        v
             NUMBER PLATE DETECTION
                        |
                        v
               PLATE REGION CROPPING
                        |
                        v
             IMAGE PREPROCESSING
                        |
                        v
                  OCR PROCESSING
                        |
                        v
             REGISTRATION NUMBER
                  EXTRACTION
                        |
                        v
               PLATE VALIDATION
                        |
                        v
                 DJANGO APPLICATION
                        |
                Search Vehicle Database
                        |
              +---------+----------+
              |                    |
              v                    v
        Vehicle Exists        New Vehicle
              |                    |
              v                    v
        Display Details      Operator Registers
              |                    |
              +---------+----------+
                        |
                        v
                OPERATOR VERIFICATION
                        |
                        v
                 CREATE VEHICLE VISIT
                        |
                        v
                  ENTRY APPROVED
```

---

# 3. Step-by-Step Number Plate Recognition Process

## Step 1: Truck Arrival and Image Capture

When a truck approaches the plant entrance, the entry camera captures one or more images of the vehicle.

An industrial IP camera can provide a video stream through RTSP.

Python and OpenCV can connect to the camera and capture frames.

Example:

```python
import cv2

camera_url = "rtsp://camera-ip/stream"

cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    raise RuntimeError("Unable to connect to camera")

ret, frame = cap.read()

if ret:
    cv2.imwrite("truck_entry.jpg", frame)

cap.release()
```

This is a simplified example. The actual capture service should handle continuous streaming, vehicle arrival detection, image selection, timestamps, and camera failures.

### Capture Considerations

- Camera should clearly capture the front or rear registration plate.
- The plate should be sufficiently large in the image.
- Avoid excessive glare and shadows.
- Capture multiple frames if the truck is moving.
- Use a suitable shutter speed to reduce motion blur.
- Preserve the original captured image.

If multiple frames are captured, the system can select the clearest frame for recognition.

---

## Step 2: Image Quality Checking

Before AI processing, the system checks whether the captured image is suitable.

Checks include:

- Image sharpness.
- Brightness and exposure.
- Motion blur.
- Plate visibility.
- Image resolution.
- Camera availability.

If the plate is blurred, too small, or obstructed, the system should attempt another capture.

OpenCV can perform basic image-quality checks.

Example:

```python
import cv2

image = cv2.imread("truck_entry.jpg")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

blur_score = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()

if blur_score < 100:
    print("Image may be blurry")
else:
    print("Image quality acceptable")
```

The threshold is illustrative and must be calibrated using actual entry-gate images.

---

## Step 3: Number Plate Detection Using YOLO

The next step is detecting the registration plate inside the truck image.

A YOLO object-detection model can be trained to recognize number plates.

The model receives the complete truck image and predicts the location of the plate.

### YOLO Detection Output

For each detected plate, YOLO returns:

1. Object class.
2. Bounding box coordinates.
3. Confidence score.

### A. Detected Object Class

The object class identifies what the model detected.

For this project, the initial dataset can contain one class:

```yaml
names:
  0: number_plate
```

The model is trained to identify the registration plate, not to read its characters.

### B. Bounding Box Coordinates

The bounding box identifies the exact location of the registration plate.

YOLO generally returns:

```text
x1, y1, x2, y2
```

Where:

- `x1`: Left boundary.
- `y1`: Top boundary.
- `x2`: Right boundary.
- `y2`: Bottom boundary.

Example:

```text
Image Resolution: 1920 × 1080

Number Plate:

x1 = 650
y1 = 420
x2 = 1100
y2 = 540
```

The system uses these coordinates to crop the plate from the original truck image.

### C. Confidence Score

The confidence score indicates how strongly the model supports the plate detection.

Example:

| Detection | Object Class | Confidence |
|---|---|---:|
| Plate 1 | number_plate | 0.98 |
| Plate 2 | number_plate | 0.91 |
| Plate 3 | number_plate | 0.62 |

A low-confidence detection may need another capture or operator verification.

A high confidence score does not guarantee that the detected region is actually a readable number plate.

### Example YOLO Inference Code

```python
from ultralytics import YOLO

model = YOLO("plate_detector.pt")

results = model.predict(
    source="truck_entry.jpg",
    conf=0.50
)

for result in results:

    for box in result.boxes:

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print({
            "class": "number_plate",
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })
```

`plate_detector.pt` contains the trained number-plate detection model.

---

## Step 4: Crop the Number Plate

After detecting the plate, the system extracts it from the original image using the bounding box coordinates.

Example:

```python
import cv2

image = cv2.imread("truck_entry.jpg")

x1, y1, x2, y2 = 650, 420, 1100, 540

plate_crop = image[
    int(y1):int(y2),
    int(x1):int(x2)
]

cv2.imwrite(
    "number_plate_crop.jpg",
    plate_crop
)
```

The cropped image contains only the registration plate.

This makes character recognition easier because OCR does not need to process the entire truck image.

---

## Step 5: Image Preprocessing

The cropped plate may contain:

- Uneven lighting.
- Shadows.
- Blur.
- Perspective distortion.
- Low contrast.
- Dirt or scratches.

OpenCV can improve the image before OCR processing.

Possible preprocessing operations:

1. Resize the plate image.
2. Convert it to grayscale.
3. Improve contrast.
4. Reduce noise.
5. Apply thresholding when appropriate.
6. Correct perspective distortion when the plate is tilted.

Example:

```python
import cv2

plate = cv2.imread("number_plate_crop.jpg")

gray = cv2.cvtColor(
    plate,
    cv2.COLOR_BGR2GRAY
)

resized = cv2.resize(
    gray,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC
)

processed = cv2.GaussianBlur(
    resized,
    (3, 3),
    0
)

cv2.imwrite(
    "processed_plate.jpg",
    processed
)
```

Preprocessing should be tested carefully. Excessive thresholding or sharpening can damage character details.

---

# 4. Extract Registration Number Using OCR

## 4.1 What Is OCR?

OCR stands for **Optical Character Recognition**.

It converts text visible in an image into machine-readable characters.

YOLO detects where the number plate is located.

OCR reads the characters printed on that plate.

For this project, PaddleOCR is a proposed starting point because it can run locally without cloud OCR APIs.

### Example

Input image:

```text
+-----------------------+
|                       |
|       WB 38 AB 1234   |
|                       |
+-----------------------+
```

OCR output:

```text
WB38AB1234
```

The extracted text is then passed to the Django application.

## 4.2 OCR Output

OCR may return:

- Recognized text.
- Text-region coordinates.
- Recognition confidence.

Example:

| Recognized Text | Confidence |
|---|---:|
| WB38AB1234 | 0.98 |
| WB38A81234 | 0.82 |
| WB38AB123 | 0.61 |

These are illustrative results.

Some characters can be confused:

| Character | Possible Confusion |
|---|---|
| 0 | O |
| 1 | I |
| 2 | Z |
| 5 | S |
| 8 | B |

The system must not blindly replace characters. It should use the expected registration format, OCR confidence, and operator verification.

## 4.3 Example PaddleOCR Code

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

results = ocr.ocr(
    "processed_plate.jpg",
    cls=True
)

for line in results:

    if line:
        for item in line:

            text = item[1][0]
            confidence = item[1][1]

            print({
                "registration_number": text,
                "confidence": confidence
            })
```

The exact initialization and inference API may vary with the installed PaddleOCR version. The implementation should be tested against the selected version.

---

## Step 5: Registration Number Validation

OCR output may contain incorrect or incomplete characters.

Therefore, the extracted registration number should be normalized and validated before searching the database.

Example:

```python
import re

def normalize_plate(text):

    cleaned = re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )

    return cleaned
```

Example:

```text
OCR Output: WB 38 AB 1234

Normalized: WB38AB1234
```

Validation should use configurable rules appropriate to the registration formats supported by the plant.

The system should check:

- Whether the extracted text is empty.
- Whether the registration format is valid.
- Whether the recognized text is sufficiently confident.
- Whether multiple conflicting plate readings exist.

An invalid or uncertain result should be sent to the gate operator for correction.

---

# 5. Django Vehicle Database Integration

After extracting and validating the registration number, the system sends the result to Django.

Django searches the vehicle master table using the normalized registration number.

### Scenario A: Vehicle Already Registered

Example:

```text
Recognized Registration: WB38AB1234

Vehicle Database:

Registration: WB38AB1234
Vehicle Type: Truck
Declared Capacity: 100 cylinders
Status: Active
```

Django displays the existing vehicle details.

The gate operator verifies that the detected registration number matches the truck.

If verified, Django creates a new vehicle visit.

### Scenario B: New Vehicle

If the registration number does not exist in the database, Django displays a registration form.

The operator can enter:

- Registration number.
- Vehicle type.
- Declared capacity, if applicable.
- Driver details, if required.
- Transport reference, if required.

After verification, the vehicle is saved in the database.

### Scenario C: Uncertain Recognition

If OCR confidence is low or the registration number is invalid:

1. Display the captured truck image.
2. Display the cropped number plate.
3. Display the recognized text.
4. Allow the operator to correct the number.
5. Save the corrected value with an audit record.

The system must not automatically create a vehicle record from an uncertain OCR result.

---

# 6. Create Vehicle Visit

Once the vehicle is identified and verified, Django creates a new vehicle visit.

A vehicle and a vehicle visit are different records.

A vehicle may enter the plant multiple times, but each entry should have its own visit record.

Example:

```text
Vehicle:
    Registration: WB38AB1234
    Vehicle Type: Truck

Vehicle Visit:
    Visit ID: VIS-2026-001
    Arrival Time: 10:30 AM
    Entry Status: APPROVED
```

This visit becomes the starting point for the loading and cylinder inspection workflow.

---

# 7. Store Recognition Results

The system should preserve the original recognition evidence.

Important information:

| Field | Description |
|---|---|
| Vehicle | Associated vehicle record |
| Vehicle Visit | Associated visit record |
| Original Image | Full truck image |
| Plate Crop | Cropped registration plate |
| Detected Text | Raw OCR output |
| Normalized Text | Cleaned registration number |
| Detection Confidence | YOLO confidence |
| OCR Confidence | Character-recognition confidence |
| Capture Timestamp | Image capture time |
| Verification Status | Pending, verified, or corrected |
| Verified By | Operator who confirmed the result |

This information supports auditing and investigation when the system reads a plate incorrectly.

---

# 8. AI Training Process

The number-plate detection model should be trained using images representative of the actual plant entrance.

The OCR engine can initially use a pretrained model. We should first evaluate whether it can read the registration plates reliably before considering custom OCR training.

## Step 1: Collect Images

Collect truck images from the actual entry camera.

Include:

- Front and rear registration plates.
- Different truck types.
- Different plate sizes.
- Daylight and night conditions.
- Different weather conditions.
- Different camera distances.
- Different camera angles.
- Blurred and partially obstructed plates.
- Dirty and damaged plates.

Use representative images from the actual plant environment.

## Step 2: Annotate Number Plates

Use CVAT or Roboflow to draw bounding boxes around registration plates.

For example:

```text
Image: truck_001.jpg

Annotation:
    Class: number_plate
    Bounding Box: [650, 420, 1100, 540]
```

The annotation tells YOLO where the registration plate is located.

## Step 3: Split the Dataset

A possible initial split:

| Dataset | Suggested Split | Purpose |
|---|---:|---|
| Training | 70% | Train the detection model |
| Validation | 20% | Tune model performance |
| Testing | 10% | Evaluate unseen images |

Split by truck or capture session to prevent near-identical images from appearing in both training and testing data.

## Step 4: Train YOLO

Start with a pretrained YOLO model and fine-tune it using the annotated number-plate dataset.

Example:

```python
from ultralytics import YOLO

model = YOLO("yolo_pretrained.pt")

model.train(
    data="plate_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0
)
```

The training settings are illustrative and should be adjusted based on the dataset and available hardware.

The trained model produces weights such as:

```text
runs/detect/train/weights/
    best.pt
    last.pt
```

`best.pt` is used for number-plate detection during inference.

## Step 5: Evaluate Detection and OCR

Evaluate the two stages separately.

### Number Plate Detection

- Precision.
- Recall.
- mAP.
- Missed-plate rate.
- False-detection rate.

### OCR Recognition

- Exact registration-number recognition rate.
- Character-level accuracy.
- Incorrect-character rate.
- Recognition confidence.
- Failure rate under poor lighting.

The most important operational metric is the percentage of vehicles for which the complete system correctly reads the registration number.

---

# 9. Proposed Django Data Model

The following models can support the vehicle-entry workflow.

| Model | Purpose |
|---|---|
| `Vehicle` | Stores vehicle registration and details |
| `VehicleVisit` | Stores each plant entry and departure |
| `Camera` | Stores entry-camera configuration |
| `PlateCapture` | Stores original truck images and plate crops |
| `PlateRecognitionResult` | Stores OCR text and confidence scores |
| `AuditLog` | Stores operator corrections and verification history |

These models can be connected to the cylinder-inspection application.

---

# 10. Important Technical Considerations

1. **Camera Position:** The entry camera must capture a sufficiently clear view of the registration plate.

2. **Motion Blur:** Moving trucks may produce blurred images. Suitable shutter speed and image capture timing are important.

3. **OCR Errors:** Similar-looking characters can cause incorrect registration numbers.

4. **Multiple Vehicles:** The system must avoid associating one vehicle's plate with another vehicle's entry event.

5. **Duplicate Entry Events:** Continuous video processing must not create multiple vehicle visits for one arrival.

6. **Uncertain Recognition:** Low-confidence results must be verified by an operator.

7. **Offline Operation:** YOLO and OCR models must be installed locally and operate without cloud APIs.

8. **Audit History:** Original images, OCR predictions, and operator corrections must be preserved.

---

# 11. Development Strategy

| Phase | Activity | Expected Result |
|---|---|---|
| 1 | Connect entry camera | Working image capture |
| 2 | Collect and annotate plate images | Training dataset |
| 3 | Train YOLO plate detector | Plate-region detection |
| 4 | Implement plate cropping and preprocessing | Clean plate images |
| 5 | Integrate OCR | Registration-number extraction |
| 6 | Implement validation and normalization | Standardized registration number |
| 7 | Integrate Django vehicle database | Existing/new vehicle identification |
| 8 | Add operator verification | Reliable vehicle registration |
| 9 | Create vehicle visit records | Complete entry workflow |
| 10 | Test under real plant conditions | Measured recognition performance |

---

# 12. Expected Final Output

For one incoming truck, the system should produce an output similar to:

```text
ENTRY INSPECTION

Capture ID: CAP-2026-001

Vehicle Registration:
    WB38AB1234

YOLO Plate Detection:
    Confidence: 0.98

OCR Recognition:
    WB38AB1234

OCR Confidence:
    0.97

Database Match:
    EXISTING VEHICLE

Vehicle Type:
    Truck

Verification:
    OPERATOR VERIFIED

Vehicle Visit:
    VIS-2026-001

Entry Status:
    APPROVED
```

The values above are illustrative.

The final objective is to create an offline vehicle-identification system that detects registration plates, reads their characters, verifies vehicles against the Django database, and creates reliable vehicle-visit records.

**Development priority:** First establish reliable plate detection and OCR using real entry-camera images. Then integrate vehicle lookup, operator verification, and visit creation into Django.
