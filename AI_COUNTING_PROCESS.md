# AI Cylinder Counting Process

## 1. Overview

The AI Cylinder Counting System uses three industrial cameras to inspect a loaded truck and estimate the total number of LPG cylinders by category.

The truck remains stationary at a predefined inspection position. The system captures images from the top, side, and rear cameras. AI processes these images, detects cylinders, identifies their types, and combines the results to calculate the number of unique physical cylinders.

The main objective is to produce an accurate cylinder count without counting the same physical cylinder multiple times across different camera views.

### Main Technologies

| Component | Technology |
|---|---|
| Backend | Python and Django |
| Database | PostgreSQL |
| Camera Integration | OpenCV and RTSP |
| Object Detection | YOLO |
| Model Training | PyTorch |
| Image Processing | OpenCV |
| Multi-Camera Association | Camera Calibration and Custom Matching Algorithms |
| Frontend | Django Templates, HTML, CSS, JavaScript |
| Reporting | PDF and CSV |

---

# 2. Overall System Workflow

The Django application manages the inspection process, while the AI processing module handles cylinder detection, classification, matching, and counting.

```text
                  DJANGO APPLICATION
                           |
                   Create Inspection
                           |
                           v
                    Capture Service
                           |
              +------------+------------+
              |            |            |
              v            v            v
             Top          Side         Rear
           Camera        Camera       Camera
              |            |            |
              +------------+------------+
                           |
                           v
                     AI PROCESSING
                           |
                     YOLO Detection
                           |
                   Type Classification
                           |
                 Cross-Camera Matching
                           |
                     Unique Counting
                           |
                           v
                      AI RESULTS
                           |
                           v
                    DJANGO DATABASE
                           |
                Compare Expected Counts
                           |
                    Approval / Review
                           |
                           v
                    Generate Report
```

### Workflow Explanation

1. The operator creates an inspection session in Django.
2. Django requests the capture service to capture images from all three cameras.
3. The capture service stores the original images and links them to the inspection session.
4. The AI processing module receives the captured images.
5. YOLO detects individual cylinders in each image.
6. The system identifies the cylinder categories.
7. Cross-camera association determines which detections represent the same physical cylinder.
8. Duplicate observations are merged.
9. The system calculates unique cylinder quantities by category.
10. Django stores the AI results in PostgreSQL.
11. AI-proposed quantities are compared with the operator-entered expected quantities.
12. The inspection is automatically approved or sent for human review.
13. Django generates the final inspection report.

---

# 3. Step-by-Step AI Counting Process

## Step 1: Capture Images from Three Cameras

Once the truck reaches the inspection position and stops, Django triggers the three cameras.

| Camera | View | Purpose |
|---|---|---|
| Camera 1 | Top / Overhead | Capture cylinder positions, rows, spacing, and arrangement |
| Camera 2 | Side | Capture cylinder bodies, stack height, and overlapping objects |
| Camera 3 | Rear | Capture rear rows and areas not clearly visible from other views |

All three images belong to the same inspection session.

### Technical Capture Process

Industrial IP cameras can provide video streams through RTSP.

Python can connect to these streams using OpenCV.

Example:

```python
import cv2

camera_url = "rtsp://camera-ip/stream"

cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    raise RuntimeError("Camera connection failed")

ret, frame = cap.read()

if ret:
    cv2.imwrite("top_camera.jpg", frame)

cap.release()
```

This is a simplified example for one camera.

The actual capture service must:

- Connect to all three cameras.
- Capture images using a common inspection trigger.
- Record capture timestamps.
- Check camera availability.
- Retry failed captures when appropriate.
- Store original images.
- Link every image to the correct inspection session.

Example capture records:

```text
Inspection ID: INS-2026-001

Top Camera:
    File: top_camera.jpg
    Capture Time: 10:30:00.100

Side Camera:
    File: side_camera.jpg
    Capture Time: 10:30:00.120

Rear Camera:
    File: rear_camera.jpg
    Capture Time: 10:30:00.110
```

The timestamps are illustrative. Production capture requires a tested synchronization mechanism.

---

## Step 2: Image Preprocessing and Quality Checking

Before sending images to the AI model, the system checks whether they are suitable for processing.

### Image Quality Checks

- Blur and sharpness.
- Brightness and exposure.
- Camera availability.
- Obstructions and visibility.
- Image resolution.
- Whether the relevant truck-bed area is visible.

OpenCV can perform basic image-quality checks.

Example: Blur detection using Laplacian variance.

```python
import cv2

image = cv2.imread("top_camera.jpg")

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

The threshold of `100` is only an example. It must be calibrated using actual plant images.

### Region of Interest (ROI)

The camera may capture areas outside the truck bed.

We can define a Region of Interest (ROI) to restrict AI processing to the relevant area.

Benefits:

- Reduces unnecessary detections.
- Avoids detecting unrelated objects.
- Reduces processing requirements.
- Improves consistency.

If an image is too blurry, dark, or obstructed, the system should request another capture instead of trusting unreliable AI results.

---

# 4. Detect Cylinders Using YOLO

## 4.1 What Is YOLO?

YOLO stands for **You Only Look Once**.

It is an object-detection model that identifies objects in an image and predicts their locations.

For this project, YOLO will be trained to identify LPG cylinders in images captured by the top, side, and rear cameras.

The model receives an image and returns information about every detected object.

For each detection, we need three important outputs:

1. Detected object class.
2. Bounding box coordinates.
3. Confidence score.

These outputs form the foundation of the cylinder-counting process.

## 4.2 Detected Object Class

The object class tells us what the model believes it has detected.

For our project, we can initially define:

```yaml
names:
  0: cylinder_14_2kg
  1: cylinder_19kg
```

Example predictions:

| Detection | Object Class |
|---|---|
| Object 1 | cylinder_14_2kg |
| Object 2 | cylinder_14_2kg |
| Object 3 | cylinder_19kg |
| Object 4 | cylinder_19kg |

This allows the system to distinguish cylinder categories.

If the visual evidence is insufficient to determine the category, the system should mark it as `UNKNOWN` rather than forcing a prediction.

The actual classes will depend on the cylinder categories used at the plant.

## 4.3 Bounding Box Coordinates

A bounding box identifies the location of a detected object inside an image.

YOLO returns coordinates representing the rectangle around the detected cylinder.

The coordinates are generally represented as:

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

Cylinder Detection:

x1 = 120
y1 = 150
x2 = 230
y2 = 390
```

The bounding box represents the region containing the detected cylinder.

We can calculate its centre:

```python
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

And its dimensions:

```python
width = x2 - x1
height = y2 - y1
```

These coordinates are useful for:

- Identifying cylinder positions.
- Comparing detections.
- Calculating relative distances.
- Estimating spatial arrangements.
- Supporting cross-camera association.

**Important:** Bounding boxes describe locations in individual camera images. Coordinates from different cameras cannot be compared directly because each camera has a different viewpoint and coordinate system.

Camera calibration and geometric transformations are needed to relate observations across views.

## 4.4 Confidence Score

The confidence score represents how strongly the model supports a particular detection.

For example:

| Detection | Predicted Class | Confidence |
|---|---|---:|
| Cylinder 1 | 14.2 kg | 0.97 |
| Cylinder 2 | 14.2 kg | 0.94 |
| Cylinder 3 | 19 kg | 0.91 |
| Cylinder 4 | 19 kg | 0.63 |

A confidence score of `0.97` indicates a strong model prediction.

A score of `0.63` indicates a less certain prediction.

We can initially define configurable confidence rules:

```text
Confidence >= 0.80:
    Accept as a high-confidence detection.

Confidence between 0.50 and 0.80:
    Keep as an uncertain candidate.

Confidence < 0.50:
    Reject or investigate based on validation rules.
```

These thresholds are examples, not final production settings.

A high confidence score does not guarantee that the detection is correct. The thresholds must be evaluated using real plant images.

## 4.5 Example YOLO Detection Output

Suppose the top camera image contains several cylinders.

The model may produce:

```text
Image: top_camera.jpg

Detection 1:
    Class: cylinder_14_2kg
    Bounding Box: [120, 150, 230, 390]
    Confidence: 0.97

Detection 2:
    Class: cylinder_14_2kg
    Bounding Box: [240, 150, 350, 390]
    Confidence: 0.94

Detection 3:
    Class: cylinder_19kg
    Bounding Box: [370, 130, 500, 410]
    Confidence: 0.91
```

These are example predictions.

The model has detected three cylinder instances in this image.

However, this is not yet the final truck count because the same cylinders may also appear in the side and rear camera images.

## 4.6 Example YOLO Inference Code

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="top_camera.jpg",
    conf=0.50
)

for result in results:

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print({
            "class_id": class_id,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })
```

`best.pt` contains the trained model weights.

We will generate this file during the model-training process.

---

# 5. Cylinder Type Classification

Detecting a cylinder and identifying its capacity are two different tasks.

For example, the system must distinguish between:

- 14.2 kg cylinders.
- 19 kg cylinders.
- Other configured cylinder categories.

### Approach A: YOLO with Multiple Classes

Train one YOLO model with different cylinder categories.

```yaml
names:
  0: cylinder_14_2kg
  1: cylinder_19kg
```

The model returns both the bounding box and predicted category.

### Approach B: Separate Detection and Classification

1. YOLO detects a cylinder.
2. The system extracts the cylinder image using its bounding box.
3. The cropped image is passed to a classification model.
4. The classification model predicts the cylinder category.

For the first prototype, Approach A is recommended if the cylinder categories can be distinguished reliably in the available camera views.

If the category cannot be determined confidently, the system should return `UNKNOWN`.

Colour alone should not be treated as a reliable cylinder identifier.

---

# 6. The Hardest Part: Avoiding Duplicate Counting

**Cross-camera association is one of the most challenging components of this project.**

The problem is that the same physical cylinder can appear in multiple camera images.

For example, consider one cylinder loaded on a truck.

It may be visible in:

- The top camera.
- The side camera.
- The rear camera.

YOLO processes each image independently.

Therefore, it may detect the same physical cylinder three times.

## 6.1 Understanding the Duplicate-Counting Problem

Imagine one physical cylinder appears in all three images.

The detection results might look like this:

| Camera | Detection ID | Predicted Class | Confidence |
|---|---|---|---:|
| Top | TOP-001 | 14.2 kg | 0.96 |
| Side | SIDE-008 | 14.2 kg | 0.93 |
| Rear | REAR-004 | 14.2 kg | 0.95 |

These are three detections, but they may represent only one physical cylinder.

If we simply add all detections:

```text
Top detections  = 1
Side detections = 1
Rear detections = 1

Total = 3
```

The system incorrectly reports three cylinders instead of one.

This is why we cannot simply add the detection counts from all three cameras.

We need to determine which detections belong to the same physical cylinder.

This process is called **Cross-Camera Association**.

## 6.2 What Is Cross-Camera Association?

Cross-camera association is the process of matching observations from different cameras that represent the same real-world object.

For our project, its responsibility is:

```text
Top Camera Detection
          |
          |
          v
   Cross-Camera Matching
          ^
          |
          |
Side Camera Detection

          +

Rear Camera Detection
          |
          v
Identify Same Physical Cylinder
          |
          v
Assign One Unique Cylinder ID
```

After matching, all observations belonging to one physical cylinder receive the same internal identity.

Example:

```text
Physical Cylinder ID: CYL-001

Associated Observations:

    Top Camera  -> TOP-001
    Side Camera -> SIDE-008
    Rear Camera -> REAR-004

Final Unique Count: 1
```

The system stores the individual observations but counts the associated physical cylinder only once.

## 6.3 How Can We Identify the Same Cylinder?

We cannot simply compare bounding box coordinates because each camera has a different viewing angle.

Instead, we need several sources of information.

### A. Camera Calibration

Camera calibration helps us understand the relationship between image coordinates and the physical inspection area.

We need to establish:

- Camera position.
- Camera orientation.
- Camera angle.
- Camera parameters.
- Relevant truck-bed dimensions.
- Relationship between camera views.

Using calibration and known reference points, we can estimate where a detected cylinder is located relative to the truck.

### B. Geometric Position Matching

Suppose the top camera detects a cylinder at a particular position on the truck bed.

The system uses the calibrated camera geometry to estimate its physical location.

It then checks whether the side and rear camera detections are geometrically consistent with that location.

For example:

```text
Physical Truck-Bed Position:

Cylinder ID: CYL-001

Estimated Position:
    X = 1.2 meters
    Y = 2.4 meters
    Z = 0.5 meters

Observations:
    Top camera: Detection 12
    Side camera: Detection 8
    Rear camera: Detection 5
```

The coordinates are illustrative.

In practice, estimating the complete three-dimensional position is more difficult than mapping a point on a flat plane.

### C. Visual Feature Matching

We can also compare the visual appearance of detected cylinders.

Possible features:

- Shape.
- Width and height proportions.
- Surface appearance.
- Visible markings.
- Distinctive visual characteristics.

An image-embedding model can convert cropped cylinder images into numerical feature vectors.

These vectors can help compare visually similar observations.

However, visual similarity alone is not sufficient. Two different cylinders may look almost identical.

### D. Neighbouring Cylinder Arrangement

The surrounding cylinders can provide additional information.

For example, if the top camera detects a cylinder between two known positions, the side and rear views can be checked against the same loading arrangement.

This information can improve matching reliability.

## 6.4 Build a Common Truck-Bed Coordinate System

A useful approach is to establish a common coordinate system for the truck bed.

Example:

```text
                    REAR
          +---------------------+
          |                     |
          |  C1   C2   C3   C4  |
          |                     |
          |  C5   C6   C7   C8  |
          |                     |
          |  C9  C10  C11  C12 |
          |                     |
          +---------------------+
                    FRONT
```

Each physical cylinder candidate can be assigned a position.

Example:

```text
CYL-001:
    Position: (X1, Y1, Z1)
    Type: 14.2 kg

CYL-002:
    Position: (X2, Y2, Z2)
    Type: 14.2 kg

CYL-003:
    Position: (X3, Y3, Z3)
    Type: 19 kg
```

The three camera views are then used to identify observations corresponding to these physical cylinder candidates.

**Important:** A two-dimensional top-view coordinate does not resolve every cylinder stacked at different heights. We may need additional depth and height information, calibrated geometry, suitable camera coverage, or assumptions about the loading arrangement.

A planar homography alone cannot solve every three-dimensional overlap problem.

## 6.5 Duplicate Removal Logic

After matching detections, we create a unique identity for each physical cylinder.

Example:

| Physical Cylinder ID | Top Detection | Side Detection | Rear Detection | Count |
|---|---|---|---|---:|
| CYL-001 | TOP-001 | SIDE-008 | REAR-004 | 1 |
| CYL-002 | TOP-002 | SIDE-009 | REAR-005 | 1 |
| CYL-003 | TOP-003 | SIDE-010 | REAR-006 | 1 |

Even though each cylinder may have three observations, each physical cylinder contributes only one to the final count.

The system should preserve all observations for traceability.

## 6.6 What Happens When Matching Is Uncertain?

Not every cylinder will be clearly visible in all camera views.

For example:

- A cylinder may be visible only from the top.
- Another cylinder may be partially hidden.
- Two detections may have similar appearance.
- A camera may fail to capture a clear image.

The system must not guess whether two detections represent the same physical cylinder.

Instead, it should assign an association status:

| Status | Meaning |
|---|---|
| `MATCHED` | Observations confidently represent one physical cylinder |
| `UNMATCHED` | No corresponding observation was found |
| `UNCERTAIN` | Multiple possible matches exist |
| `REVIEW_REQUIRED` | Human verification is needed |

An uncertain association should be preserved for review instead of silently merging or counting observations.

## 6.7 Why This Component Is Difficult

| Challenge | Problem |
|---|---|
| Different camera angles | The same cylinder looks different in different views |
| Occlusion | Some cylinders are partially or completely hidden |
| Similar appearance | Different cylinders can look almost identical |
| Stacked arrangement | Cylinders may overlap in three-dimensional space |
| Calibration errors | Camera movement can affect geometric matching |
| Incorrect association | Two different cylinders may be merged into one |
| Missed detections | A cylinder may not be detected in one or more views |

The goal is to minimize both duplicate counts and missed cylinders.

This component will require separate development and real-world testing. YOLO detection alone does not solve cross-camera association.

---

# 7. Unique Counting and Count Calculation

After cross-camera association, the system has a list of unique physical-cylinder candidates.

Each candidate contributes only once to the final count.

Example:

| Cylinder Category | Unique Count |
|---|---:|
| 14.2 kg | 80 |
| 19 kg | 20 |
| **Total** | **100** |

The total count is calculated as:

```python
total_count = sum(
    cylinder_counts.values()
)
```

Example:

```python
cylinder_counts = {
    "14.2kg": 80,
    "19kg": 20
}

total_count = sum(cylinder_counts.values())

print(total_count)
```

Output:

```text
100
```

Only sufficiently reliable, unique cylinder candidates should contribute to an automatically approved result.

Uncertain candidates must be accounted for separately and may require review.

---

# 8. Compare AI Count with Expected Quantity

Before inspection, the loading operator enters the expected cylinder quantities into Django.

The system compares the expected quantities with the AI-proposed quantities.

Example:

| Category | Expected | AI Count | Difference |
|---|---:|---:|---:|
| 14.2 kg | 80 | 80 | 0 |
| 19 kg | 20 | 20 | 0 |
| **Total** | **100** | **100** | **0** |

The system should preserve three separate quantities:

- Expected count: Entered by the operator.
- AI-proposed count: Calculated by the AI system.
- Verified count: Accepted after human review, when applicable.

These values must never silently overwrite one another.

---

# 9. Inspection Decision

After counting, the system determines the inspection status.

| Status | Meaning |
|---|---|
| `AUTO_APPROVED` | Required cameras and quality checks pass, counts satisfy configured rules, and no unresolved uncertainty remains |
| `REVIEW_REQUIRED` | Count mismatch, uncertain cylinder type, possible duplicate, hidden cylinder, or unresolved association |
| `RECAPTURE_REQUIRED` | Camera failure, blurred image, poor lighting, or obstructed capture |

Automatic approval must consider more than YOLO confidence.

It should also consider:

- Image quality.
- Camera availability.
- Camera calibration.
- Cylinder detection confidence.
- Cylinder classification confidence.
- Cross-camera association certainty.
- Count reconciliation.
- Unresolved detections.

Human review should be required for exceptions rather than every truck.

Automatic approval should be enabled only after successful real-world validation.

---

# 10. Store Results in Django and PostgreSQL

After AI processing, Django stores the results in PostgreSQL.

The database should preserve the complete inspection history.

### Important Data

| Data | Description |
|---|---|
| Vehicle | Vehicle registration and details |
| Inspection Session | Inspection reference and status |
| Camera Capture | Original image and capture timestamp |
| AI Detection | Object class, bounding box, confidence |
| Cylinder Association | Cross-camera observations linked to one physical cylinder |
| Cylinder Count Result | Unique AI count by category |
| Review Decision | Human review and approval information |
| Count Revision | Original and corrected quantities |
| Inspection Report | Generated report reference |

### Example AI Detection Record

```text
Detection ID: DET-001
Inspection ID: INS-2026-001
Camera: TOP

Class: cylinder_14_2kg
Bounding Box: [120, 150, 230, 390]
Confidence: 0.97

Association ID: CYL-001
```

### Example Cylinder Association Record

```text
Cylinder ID: CYL-001

Observations:
    DET-001 (TOP)
    DET-008 (SIDE)
    DET-004 (REAR)

Category: 14.2 kg
Association Status: MATCHED
```

The original camera images, AI predictions, model version, and operator corrections must be retained for audit purposes.

---

# 11. AI Training Process

The cylinder detection model must be trained using images representative of the actual plant environment.

## Step 1: Collect Images

Collect top, side, and rear images of loaded trucks under different conditions.

Include:

- Different cylinder categories.
- Different truck sizes.
- Different loading arrangements.
- Different numbers of cylinders.
- Overlapping cylinders.
- Partially hidden cylinders.
- Daylight and night conditions.
- Different lighting conditions.
- Different camera distances and angles.

All three camera images of the same truck load should be grouped under one load-session ID.

Example:

```text
dataset/
│
├── truck_001/
│   ├── top.jpg
│   ├── side.jpg
│   └── rear.jpg
│
├── truck_002/
│   ├── top.jpg
│   ├── side.jpg
│   └── rear.jpg
│
└── truck_003/
    ├── top.jpg
    ├── side.jpg
    └── rear.jpg
```

## Step 2: Annotate Images

Use CVAT or Roboflow to draw bounding boxes around cylinders and assign their categories.

Example classes:

```yaml
names:
  0: cylinder_14_2kg
  1: cylinder_19kg
```

Each annotated cylinder should have:

- Bounding box.
- Cylinder category, if identifiable.
- Appropriate visibility and difficulty labels, if supported by the annotation workflow.

## Step 3: Split the Dataset

Divide the dataset into:

| Dataset | Suggested Split | Purpose |
|---|---:|---|
| Training | 70% | Train the model |
| Validation | 20% | Evaluate and tune training |
| Testing | 10% | Evaluate unseen examples |

Split by truck/load session rather than individual images.

This prevents images of the same physical load from appearing in both training and testing data.

## Step 4: Train YOLO

Start with a pretrained YOLO model and fine-tune it using the annotated cylinder dataset.

Example:

```python
from ultralytics import YOLO

model = YOLO("yolo_pretrained.pt")

model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0
)
```

The settings are illustrative and should be adjusted according to the dataset, available GPU memory, and training results.

During training:

1. The model receives an image.
2. It predicts cylinder locations and categories.
3. Predictions are compared with manually annotated labels.
4. Training calculates prediction errors.
5. Model weights are updated through backpropagation.
6. This process repeats over multiple training epochs.

The trained model produces weights such as:

```text
runs/detect/train/weights/
    best.pt
    last.pt
```

`best.pt` is the checkpoint selected using validation performance.

## Step 5: Evaluate the Model

Evaluate:

- Detection precision.
- Detection recall.
- Mean Average Precision (mAP).
- Cylinder classification accuracy.
- Missed-cylinder rate.
- False-detection rate.
- Unique cylinder-count accuracy.
- Cross-camera association accuracy.
- Duplicate-count rate.
- Processing time per inspection.

The final unique cylinder-count accuracy is more important to the business process than detection accuracy alone.

---

# 12. Proposed AI Components

The initial AI system can be divided into three major components.

### Component 1: Cylinder Detection Model

**Technology:** YOLO + PyTorch

Responsibilities:

- Detect cylinders in each camera image.
- Generate bounding boxes.
- Predict cylinder categories.
- Return confidence scores.

### Component 2: Cross-Camera Association

**Technology:** OpenCV calibration, geometric reasoning, visual matching, and custom algorithms.

Responsibilities:

- Relate observations from different camera views.
- Estimate physical cylinder positions.
- Associate detections representing the same physical cylinder.
- Prevent duplicate counting.
- Flag uncertain associations.

This component may combine traditional computer vision and learned features. It does not necessarily require a separate neural network.

### Component 3: Image Quality Checking

**Technology:** OpenCV initially.

Responsibilities:

- Check blur.
- Check brightness.
- Detect unsuitable images.
- Identify failed or missing camera captures.
- Request recapture when necessary.

A separate trained neural network is not required for the first version.

---

# 13. Development Strategy

The project should be developed incrementally.

| Phase | Development Activity | Expected Result |
|---|---|---|
| 1 | Build Django vehicle and inspection workflow | Working application foundation |
| 2 | Integrate three cameras and capture images | Linked camera evidence |
| 3 | Collect and annotate cylinder images | Training dataset |
| 4 | Train YOLO cylinder detector | Cylinder detection |
| 5 | Implement cylinder classification | Category predictions |
| 6 | Develop camera calibration and cross-camera association | Unique physical-cylinder candidates |
| 7 | Implement duplicate removal and counting | Unique counts by category |
| 8 | Compare expected and AI counts | Inspection decision |
| 9 | Add review and report generation | Complete inspection workflow |
| 10 | Perform shadow-mode testing | Measured real-world performance |

---

# 14. Important Technical Considerations

1. **Occlusion:** Some cylinders may be hidden behind others. The system must identify uncertain cases rather than inventing counts.

2. **Duplicate Counting:** The same physical cylinder may appear in multiple camera views. Cross-camera association must happen before calculating the final count.

3. **Cylinder Classification:** Different categories must be identified using reliable visual evidence. Uncertain classifications should be flagged.

4. **Camera Calibration:** Camera positions must remain fixed and calibrated. Movement may affect geometric matching.

5. **Image Quality:** Automatic approval must be blocked when required images are unsuitable.

6. **Association Reliability:** Two different cylinders must not be merged merely because they look similar.

7. **Real-World Validation:** AI counts must initially be compared against independently verified manual counts.

8. **Offline Processing:** Model weights, Python dependencies, image processing, and inference must be available locally without cloud AI APIs.

---

# 15. Expected Final Output

For one completed truck inspection, the system should produce an output similar to:

```text
Inspection ID: INS-2026-001
Vehicle: WB-XX-1234

Expected Quantities:
14.2 kg: 80
19 kg: 20

AI-Detected Unique Quantities:
14.2 kg: 80
19 kg: 20

Total Expected: 100
Total AI Count: 100

Difference: 0

Image Quality: ACCEPTABLE
Cross-Camera Association: COMPLETED
Unresolved Detections: 0

Inspection Status: AUTO_APPROVED
```

The values above are illustrative.

The final objective is to build a reliable, offline AI-based inspection system that detects cylinders from multiple camera views, identifies their categories, prevents duplicate counting, calculates unique quantities, and maintains complete digital inspection records.

**Development priority:** First prove that YOLO can reliably detect cylinders in individual camera images. Then develop and validate cross-camera association, because that is the most technically challenging part of obtaining an accurate final truck count.
