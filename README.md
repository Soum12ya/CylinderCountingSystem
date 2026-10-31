# AI-Based Cylinder Counting & Inspection System

> **Offline AI-powered industrial platform for truck registration,
> multi-camera cylinder counting, safety-cap verification, and digital
> inspection reporting.**

**Stack:** Python · Django · PostgreSQL · OpenCV · YOLO · Industrial IP
Cameras\
**Deployment:** On-premises / offline-first

------------------------------------------------------------------------

## 1. Introduction

The AI-Based Cylinder Counting & Inspection System is designed to
digitize the inspection of trucks carrying LPG cylinders. It combines
industrial cameras, computer vision, and a Django application to
identify vehicles, inspect loaded cylinders, estimate quantities, verify
safety-cap presence, and preserve image evidence.

### Objectives

-   Identify and register trucks at the entry gate.
-   Record expected cylinder quantities before inspection.
-   Capture loaded trucks from three angles: **top, side, and rear**.
-   Detect cylinders and estimate their types and quantities.
-   Associate detections across camera views to avoid counting the same
    physical cylinder more than once.
-   Detect safety-cap presence where the images support reliable
    verification.
-   Compare AI results with operator-entered expected quantities.
-   Automatically approve clear, high-confidence inspections.
-   Send uncertain or mismatched inspections for human review.
-   Generate searchable reports with image evidence and an audit trail.
-   Operate locally without depending on cloud AI APIs.

> **Scope clarification:** The supplied reference also describes
> scan-based unloading, where a cylinder is counted when it crosses a
> virtual scan line. This README focuses on the current project
> direction: static three-view inspection of a loaded truck. Scan-line
> unloading can be developed later as a separate operating mode.

------------------------------------------------------------------------

## 2. Operational Workflow

### 2.1 Truck entry and registration

1.  A truck arrives at the entry gate.
2.  The entry camera captures one or more frames of the registration
    plate.
3.  AI detects the plate and reads the registration number using OCR.
4.  The application searches the vehicle master.
5.  If the vehicle is unknown or the plate reading is uncertain, the
    gate operator verifies or registers it.
6.  Django creates a vehicle-visit record.
7.  The authorized operator records the entry.

### 2.2 Expected loading registration

Before inspection, the operator records the expected quantity by
cylinder category.

| Cylinder Category  | Expected Quantity |
|:-------------------|------------------:|
| 14.2 kg            | 80                |
| 19 kg              | 20                |
| **Total**          | **100**           |

Expected quantities are operational inputs. They must remain separate
from AI-estimated and operator-verified quantities.

### 2.3 Three-camera loaded-truck inspection

After loading, the truck moves to a fixed inspection position and stops.
The application triggers three synchronized cameras.

  | Camera | View | Purpose |
|:-------|:-----|:--------|
| Camera 1 | Top / overhead | Detect cylinder positions, rows, spacing, and arrangement. |
| Camera 2 | Side (left or right) | Inspect cylinder bodies, stack height, overlap, and partially hidden objects. |
| Camera 3 | Rear of truck | Inspect rear rows and areas not clearly visible from the other views. |

All captures are linked to the same truck visit and inspection session.

**Capture requirements**

-   The truck must be stationary at the configured inspection position.
-   Camera timestamps should be synchronized.
-   The relevant truck-bed area should be visible.
-   Check sharpness, exposure, obstruction, and camera availability
    before trusting the images.
-   Preserve original captures; preprocessing must not replace the
    originals.
-   Never commit camera credentials or private stream URLs to source
    control.

### 2.4 AI counting and reconciliation

1.  **Image preparation:** apply camera-specific calibration, region
    selection, and image-quality checks.
2.  **Cylinder detection:** detect visible cylinder instances in each
    camera image.
3.  **Cross-view association:** determine which detections from
    different views refer to the same physical cylinder.
4.  **Duplicate suppression:** merge matching observations so one
    physical cylinder contributes only once to the unique count.
5.  **Type classification:** estimate the cylinder category, such as
    14.2 kg or 19 kg, when sufficient visual evidence exists.
6.  **Cap inspection:** detect visible safety caps and mark uncertain or
    obstructed cases for review.
7.  **Count calculation:** produce a unique count by cylinder category.
8.  **Reconciliation:** compare AI-proposed quantities with
    operator-entered expected quantities.
9.  **Decision:** automatically approve only when configured quality and
    confidence checks pass; otherwise request review or recapture.

> A cylinder may appear in two or three camera images. Simply adding
> detections will overcount. Camera calibration and geometric reasoning
> are required; a planar homography alone does not solve every 3D
> overlap or stacking situation.

### 2.5 Inspection outcomes

-   `AUTO_APPROVED` --- required cameras and quality checks pass, counts
    satisfy configured rules, and no unresolved uncertainty remains.
-   `REVIEW_REQUIRED` --- count mismatch, uncertain cylinder
    identity/type, possible duplicate, hidden cylinder, cap uncertainty,
    or unusual loading arrangement.
-   `RECAPTURE_REQUIRED` --- a camera is unavailable or an image is
    blurred, obstructed, too dark, or otherwise unsuitable.

Human approval is **exception-based**, not mandatory for every truck.
Thresholds must be validated against real plant data before automatic
approval is enabled in production.

------------------------------------------------------------------------

## 3. Camera Placement

Camera placement must be confirmed through a site survey.

| Camera | Recommended Placement | Role |
|:-------|:----------------------|:-----|
| Top camera | Fixed overhead position covering the truck bed | Top-view inspection |
| Side camera | Fixed left or right side position | Cylinder bodies and stack geometry |
| Rear camera | Behind the truck, facing the loaded bed | Rear-row and arrangement inspection |
| Entry camera | At the vehicle entry gate | Registration-plate capture |

### Installation considerations

-   Select industrial cameras and lenses suitable for mounting distance
    and field of view.
-   Provide adequate lighting for day and night operation.
-   Consider glare, shadows, dust, vibration, weather protection, and
    compression.
-   Synchronize camera time and capture triggers.
-   Keep camera mounts fixed and recalibrate after movement.
-   Monitor camera health; block automatic approval when a required
    camera fails.
-   Confirm hazardous-area and electrical installation requirements with
    plant safety and engineering teams.

Exact camera models, resolution, frame rate, lenses, lighting, and
mounting dimensions remain site-survey decisions.

------------------------------------------------------------------------

## 4. AI Vision Pipeline

### 4.1 Vehicle number-plate recognition

The entry-gate pipeline should detect the plate region, correct
perspective where practical, run OCR, and return a candidate
registration number with confidence information. Operators must be able
to correct uncertain readings.

### 4.2 Cylinder detection

A YOLO-family object detector is a proposed starting point for detecting
cylinders in top, side, and rear images. Each detection should retain
its camera/capture reference, bounding box, predicted class, confidence,
and model version.

### 4.3 Multi-camera association

The same cylinder may appear in multiple camera images. The system must
associate these observations before producing a unique count.

Potential signals include:

-   Camera calibration and known camera positions.
-   Relative cylinder positions and arrangement.
-   Geometric constraints from the truck bed.
-   Visual features and appearance descriptors.
-   Consistency across the three views.

If the system cannot confidently decide whether two detections represent
the same cylinder, it should preserve the uncertainty and request review
instead of guessing.

### 4.4 Cylinder type classification

Classify cylinder categories only when image evidence is sufficient.
Possible signals include visible markings, dimensions, shape, and
validated visual features. Colour alone should not be treated as a
reliable identifier. Uncertain cases should remain unknown or
review-required.

### 4.5 Safety-cap detection

Possible outcomes:

-   Cap present.
-   Cap missing.
-   Cap state uncertain / not visible.

A cap must not be marked missing merely because it is hidden, blurred,
or outside the camera view.

### 4.6 Confidence and quality checks

Automatic approval should consider more than one model confidence score:

-   Image quality for every required camera.
-   Camera availability and calibration status.
-   Cylinder detection confidence.
-   Cross-camera association certainty.
-   Type-classification certainty.
-   Cap-detection certainty, where required.
-   Difference between expected and AI-proposed quantities.
-   Unusual or unsupported loading arrangements.

Thresholds and tolerances must be configurable and validated against
representative test data.

------------------------------------------------------------------------

## 5. Counting Rules and Data Integrity

Keep each quantity explicit; do not store every value in one ambiguous
`count` field.

 | Field | Meaning |
|:------|:--------|
| `expected_count` | Operator-entered quantity before inspection |
| `ai_proposed_count` | Unique cylinder quantity estimated by AI |
| `ai_count_by_type` | AI-proposed quantity for each cylinder category |
| `cap_detected_count` | Cylinders for which a cap was positively detected |
| `cap_missing_count` | Cylinders for which a missing cap was positively detected |
| `cap_uncertain_count` | Cylinders whose cap state could not be determined |
| `verified_count` | Quantity accepted after human review, if required |
| `count_variance` | Difference between AI-proposed and expected quantity |
| `inspection_status` | Current inspection decision/status |

### Data integrity rules

-   Never silently overwrite an AI estimate when an operator corrects
    it.
-   Store the original AI result, corrected result, reason, user, and
    timestamp.
-   Preserve original camera images and link them to the inspection
    session.
-   Store model and camera-calibration versions used for every
    inspection.
-   Keep expected, AI-proposed, and operator-verified quantities
    distinct.
-   Make processing retryable without creating duplicate sessions or
    count events.
-   Do not automatically approve an inspection with missing required
    camera evidence.

------------------------------------------------------------------------

## 6. Risk-Based Human Review

Human review is for exceptions, not every truck.

| Condition | System Action |
|:----------|:--------------|
| Expected and AI counts match; all checks pass | Eligible for automatic approval |
| Count differs from expected quantity | `REVIEW_REQUIRED` |
| Cylinder identity across views is uncertain | `REVIEW_REQUIRED` |
| Cylinder type is uncertain | `REVIEW_REQUIRED` |
| Cap is possibly missing or cannot be verified | Apply configured cap policy; review uncertain cases |
| Unusual loading pattern | `REVIEW_REQUIRED` |
| Image is blurred, dark, or obstructed | Retry capture, then `RECAPTURE_REQUIRED` |
| Required camera is disconnected | Block automatic approval |
| Camera moved or calibration is invalid | Block automatic approval until checked |

The review screen should show truck details, expected and AI counts, all
three images, highlighted uncertain detections, cap results, and the
reason for the review flag. Operators can confirm, correct, request
recapture, or escalate. A correction should require a reason and create
an audit entry.

------------------------------------------------------------------------

## 7. Inspection Report Generation

A completed inspection report should contain:

-   Truck registration, type, and declared capacity.
-   Visit and inspection-session references.
-   Expected quantity by cylinder category.
-   AI-proposed unique count by category.
-   Operator-verified count, if applicable.
-   Cap present, missing, and uncertain quantities.
-   Top, side, rear, and entry-gate evidence references, where
    available.
-   Capture date and time.
-   AI model and camera-calibration versions.
-   Inspection status and discrepancy flags.
-   Review history, correction reasons, and approving user.

Reports should be searchable and exportable in PDF, CSV, and XLSX
formats, subject to implementation and operational requirements.

------------------------------------------------------------------------

## 8. System Architecture

``` text
                 INDUSTRIAL CAMERA LAYER
       ┌─────────────────────────────────────────┐
       │ Entry camera                            │
       │ Top camera | Side camera | Rear camera  │
       └────────────────────┬────────────────────┘
                            │ RTSP / ONVIF / trigger
                            ▼
                  CAPTURE & INGESTION
       ┌─────────────────────────────────────────┐
       │ Camera capture service                  │
       │ Timestamp synchronization               │
       │ Image-quality checks                    │
       │ Original evidence storage               │
       └────────────────────┬────────────────────┘
                            ▼
                    OFFLINE AI PIPELINE
       ┌─────────────────────────────────────────┐
       │ Number-plate detection + OCR            │
       │ Cylinder detection                      │
       │ Multi-camera association                │
       │ Duplicate suppression                   │
       │ Cylinder type classification            │
       │ Safety-cap detection                    │
       │ Confidence and quality evaluation       │
       │ Count reconciliation                    │
       └────────────────────┬────────────────────┘
                            ▼
                    DJANGO APPLICATION
       ┌─────────────────────────────────────────┐
       │ Vehicle master and visit management     │
       │ Expected loading registration           │
       │ Inspection sessions                     │
       │ AI result and exception handling        │
       │ Risk-based review console               │
       │ Reports and audit logs                  │
       │ User roles and permissions              │
       └──────────────┬──────────────┬───────────┘
                      │              │
                      ▼              ▼
                 POSTGRESQL      EVIDENCE STORAGE
                 + pgvector      Local disk / NAS
                 Operational     Original images
                 records         and derived crops
                      │
                      ▼
                 REPORTING / EXPORT
                 PDF · CSV · XLSX
```

### Architecture layers

1.  **Camera layer** --- entry-gate and top, side, and rear inspection
    cameras.
2.  **Ingestion layer** --- capture frames, synchronize timestamps,
    check image quality, and store original evidence.
3.  **AI layer** --- plate OCR, cylinder detection, multi-view
    association, type classification, cap detection, and count
    reconciliation.
4.  **Application layer** --- Django workflows for vehicles, visits,
    loading records, inspections, review, and reporting.
5.  **Data layer** --- PostgreSQL for structured records; pgvector where
    embeddings are useful; local/NAS storage for original evidence.
6.  **Presentation layer** --- Django Templates with HTML, CSS, and
    vanilla JavaScript.

------------------------------------------------------------------------

## 9. Technology Stack

  | Layer | Proposed Technology |
|:------|:-------------------|
| Language | Python 3.11+ |
| Backend | Django |
| Optional API | Django REST Framework |
| Frontend | Django Templates, HTML, CSS, vanilla JavaScript |
| Database | PostgreSQL 16+ |
| Vector search | pgvector (optional) |
| Computer vision | OpenCV |
| Object detection | YOLO-family model (offline inference) |
| OCR | PaddleOCR or another locally deployable OCR engine |
| Deep learning | PyTorch |
| Camera integration | RTSP / ONVIF, subject to camera support |
| Background jobs | Django management-command worker initially; Celery + Redis if needed |
| Reporting | WeasyPrint or equivalent PDF generation; CSV/XLSX export |
| Production server | Gunicorn + Nginx |
| Testing | pytest / Django test framework |

> This is a proposed baseline. Pin compatible package versions after
> validating the target operating system, camera SDKs, inference
> runtime, and deployment hardware.

### Why PostgreSQL and pgvector?

PostgreSQL stores vehicles, visits, cameras, captures, detections,
counts, reviews, and reports. pgvector is optional and can support
image-embedding similarity search, retrieval of visually similar
exceptions, and near-duplicate evidence investigation.

Embeddings complement---not replace---camera geometry, tracking, and
explicit multi-view association. pgvector is not itself a
cylinder-counting algorithm.

------------------------------------------------------------------------
## 10. User Interface Preview

The system provides a simple and intuitive dashboard for managing vehicle
entry, loading operations, AI-based cylinder inspection, and inspection reports.

### 1. Vehicle Entry – Number Plate Detection

The entry camera captures the incoming truck. The system uses AI-based
number plate detection and OCR to identify the vehicle registration number.
The operator can verify the detected information before proceeding.


### 2. Post-Loading – Multi-Camera AI Inspection

After loading, the system captures images of the truck using three
synchronized cameras: top, side, and rear views.

The AI pipeline analyzes the captured images, identifies cylinders,
associates observations across camera views, and calculates unique
cylinder counts by category.


### 3. Generated Inspection Report

After inspection, the system generates a digital report containing
vehicle details, visit information, expected quantities, AI-proposed
counts, count discrepancies, inspection status, and captured image evidence.

Reports can be exported in PDF and CSV formats.

![Generated Inspection Report](https://github.com/user-attachments/assets/afecea8f-7279-4266-ad59-f6dfc4576153)

------------------------------------------------------------------------

## 11. Proposed Django Data Model

 | Model | Purpose / Example Fields |
|:------|:-------------------------|
| `Vehicle` | Registration number, vehicle type, declared capacity, active status |
| `VehicleVisit` | Vehicle, visit reference, arrival/departure time, visit status |
| `LoadingRecord` | Visit, operator, loading start/end, loading status |
| `ExpectedCylinderItem` | Loading record, cylinder type, expected quantity |
| `Camera` | Camera code, view type, required flag, stream reference, calibration |
| `InspectionSession` | Visit, loading record, status, model version, expected and AI totals |
| `CameraCapture` | Session, camera, capture timestamp, image reference, quality status |
| `AIDetection` | Capture, class, bounding box, confidence, tracking reference |
| `CylinderAssociation` | Cross-camera observations associated with one physical cylinder |
| `CylinderCountResult` | Session, cylinder type, AI-proposed quantity |
| `CapInspectionResult` | Associated cylinder, cap state, confidence |
| `ReviewDecision` | Session, reviewer, decision, correction reason, timestamp |
| `CountRevision` | Previous value, corrected value, reason, changed by |
| `InspectionReport` | Session, report status, generated time, file reference |
| `AuditLog` | User, action, entity, timestamp, details |

Use database constraints and transactions to prevent invalid state
transitions and duplicate processing. Refine the schema as the workflows
and camera data are validated.

------------------------------------------------------------------------

## 12. Suggested Project Structure

``` text
cylinder_inspection/
├── README.md
├── manage.py
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── apps/
│   ├── accounts/              # Authentication, roles, permissions
│   ├── vehicles/              # Vehicle master and registration
│   ├── visits/                # Gate entry and visit lifecycle
│   ├── loading/               # Expected cylinder quantities
│   ├── cameras/               # Camera configuration and calibration
│   ├── captures/              # Image capture and evidence records
│   ├── inspections/           # Inspection session lifecycle
│   ├── ai_results/            # Detections, associations, classifications
│   ├── approvals/             # Automatic decisions and human review
│   ├── reports/                # PDF, CSV, XLSX reports
│   └── audit/                  # Activity history
├── ai/
│   ├── plate_recognition/
│   ├── cylinder_detection/
│   ├── cylinder_classification/
│   ├── cap_detection/
│   ├── multi_camera_association/
│   ├── counting/
│   ├── calibration/
│   └── evaluation/
├── model_weights/              # Local model files; usually git-ignored
├── templates/                  # Django HTML templates
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── media/                      # Local evidence; git-ignored
├── workers/                    # Background processing commands
├── tests/
│   ├── unit/
│   ├── integration/
│   └── ai_evaluation/
└── deployment/
    ├── nginx/
    ├── gunicorn/
    └── backup/
```

This is a suggested structure and can be adjusted to match the actual
repository.

------------------------------------------------------------------------

## 13. Dataset Collection and Model Evaluation

Real plant images are essential. Models trained only on internet images
or a small set of clean photographs may not be reliable in the loading
environment.

### Data to collect

-   Top, side, and rear images of the same truck/load.
-   Different truck sizes and loading arrangements.
-   Different cylinder categories and mixed loads.
-   Multiple rows and stack heights.
-   Daylight, night, shadows, glare, and changing illumination.
-   Partial occlusion and overlapping cylinders.
-   Dirty, damaged, or unusual cylinders.
-   Caps present, missing, obscured, and uncertain.
-   Blurred, compressed, and obstructed captures for quality testing.

### Annotation

CVAT or Roboflow can be used to annotate cylinder bounding boxes,
visually identifiable cylinder categories, cap state, difficult/occluded
examples, and cross-camera identity for selected synchronized sessions.

### Dataset splitting

Split training, validation, and test data by **truck/load session**, not
by randomly distributing adjacent frames. Images of the same physical
load should not leak into both training and test sets.

### Evaluation metrics

-   Detection precision, recall, and mAP.
-   Per-category classification accuracy.
-   Unique cylinder-count exact-match rate.
-   Count error distribution.
-   Cross-camera association accuracy.
-   Duplicate and missed-cylinder rates.
-   Cap-detection precision and recall.
-   Image-quality rejection accuracy.
-   Human-review rate.
-   Average processing time per inspection.

The supplied reference suggests approximately 1,500--2,000 images per
truck type as a planning estimate. Actual dataset needs depend on
diversity and measured performance; image quantity alone does not
guarantee accuracy.

------------------------------------------------------------------------

## 14. Getting Started

The following commands are a starting point and assume that the Django
project and dependency files have been created.

### 14.1 Clone the repository

``` bash
git clone <your-repository-url>
cd cylinder_inspection
```

### 14.2 Create a virtual environment

**Windows**

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 14.3 Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 14.4 Configure environment variables

Create a local `.env` file based on `.env.example`.

``` env
DJANGO_SECRET_KEY=replace-with-a-secure-local-secret
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost

DATABASE_NAME=cylinder_inspection
DATABASE_USER=postgres
DATABASE_PASSWORD=replace-me
DATABASE_HOST=127.0.0.1
DATABASE_PORT=5432

EVIDENCE_ROOT=./media/evidence
MODEL_ROOT=./model_weights
```

Do not commit `.env`, production secrets, camera passwords, or private
stream URLs.

### 14.5 Prepare PostgreSQL

``` sql
CREATE DATABASE cylinder_inspection;
```

If pgvector is implemented and installed on the database server:

``` sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 14.6 Run migrations and create an administrator

``` bash
python manage.py migrate
python manage.py createsuperuser
```

### 14.7 Start the development server

``` bash
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

Camera capture, AI inference, and background-worker commands depend on
the actual implementation and should be documented when those modules
are added.

------------------------------------------------------------------------

## 15. Offline Deployment

The target deployment is an on-premises plant server or edge
workstation.

-   Run Django, the database, AI inference, and evidence storage inside
    the plant network.
-   Keep required model weights and Python packages available locally.
-   Do not require cloud APIs for plate OCR, cylinder detection,
    classification, or cap inspection.
-   Restrict access to the camera network and application server.
-   Store evidence on local disk or a managed NAS with backup and
    retention policies.
-   Monitor disk capacity, database health, camera availability, and
    worker status.
-   Test the complete workflow with external internet access
    disconnected.

Hardware sizing depends on camera resolution, simultaneous streams,
model size, processing-time targets, and truck throughput. Confirm
hardware through representative performance tests.

------------------------------------------------------------------------

## 16. Security, Roles, and Audit

 | Role | Responsibility |
|:-----|:--------------|
| Admin | User management, system configuration, camera/model settings |
| Gate operator | Vehicle identification, registration, and visit entry |
| Loading operator | Expected quantity entry and loading workflow |
| Inspector | Review exceptions and correct inspection results |
| Supervisor | Approve escalated cases and review discrepancies |
| Auditor | Read-only access to reports and audit history |

Security requirements:

-   Use Django authentication and role-based permissions.
-   Keep secrets and camera credentials out of source control.
-   Restrict access to original evidence and exported reports.
-   Preserve AI results and operator revisions.
-   Record who performed each important action and when.
-   Back up the database and evidence storage.
-   Test restore procedures.
-   Log processing failures and make safe retries possible.

------------------------------------------------------------------------

## 17. Implementation Roadmap

 | Phase | Main Work | Deliverable |
|:------|:----------|:------------|
| 1. Requirements and site survey | Confirm truck types, cylinder categories, process, camera positions, and acceptance rules | Approved workflow and hardware requirements |
| 2. Django foundation | Project setup, PostgreSQL, authentication, roles, and audit foundation | Working application skeleton |
| 3. Vehicle and visit management | Vehicle master, entry-gate registration, visit lifecycle | Recorded truck visits |
| 4. Loading registration | Expected quantities by cylinder type | Loading records |
| 5. Camera integration | Entry, top, side, rear capture; synchronization and quality checks | Linked image evidence |
| 6. Dataset preparation | Collect, label, and split real plant images | Versioned dataset |
| 7. AI detection | Cylinder, plate, and cap detection models | Per-image detections |
| 8. Multi-camera association | Calibration, cross-view matching, duplicate suppression | Unique cylinder candidates |
| 9. Reconciliation and review | Compare expected and AI counts; implement exception-based review | Inspection decisions |
| 10. Reports and audit | Generate reports and searchable history | Digital inspection records |
| 11. Shadow-mode validation | Compare AI results with independently verified manual counts | Measured accuracy and review rate |
| 12. Controlled deployment | Enable automatic approval only for validated cases | Production pilot |

Timeline depends on site access, camera procurement, dataset
availability, model performance, and integration requirements. Commit
dates only after the site survey and pilot plan.

------------------------------------------------------------------------

## 18. Testing and Acceptance

### Functional tests

-   Vehicle plate detected and operator correction supported.
-   A truck visit is created only once for the relevant entry event.
-   Expected quantities are stored by cylinder category.
-   Required camera images are linked to the correct visit/session.
-   Duplicate processing does not create duplicate counts.
-   AI and operator-verified values remain separate.
-   Manual corrections require a reason and create an audit record.
-   Reports show evidence, counts, and inspection status.

### AI and operational tests

-   Count accuracy across different truck and loading arrangements.
-   Performance under low light, glare, shadows, and partial occlusion.
-   Cross-camera duplicate matching and missed-cylinder behavior.
-   Cylinder-type and cap-detection performance.
-   Camera failure and recapture behavior.
-   Processing time under expected operating load.
-   Offline operation and recovery after restart.

### Rollout recommendation

Begin in **shadow mode**: AI produces counts, but the independently
verified operator count remains authoritative. Compare results across
representative real truck visits, investigate mismatches, and tune
models and thresholds. Enable automatic approval only for scenarios that
meet acceptance criteria agreed with the plant.

------------------------------------------------------------------------

## 19. Open Questions and Site Inputs

 | Item | Status |
|:-----|:-------|
| Truck types and declared capacities | To confirm |
| Cylinder categories and mixed-load rules | To confirm |
| Exact loading and inspection positions | Site survey required |
| Camera make, resolution, lens, FPS, and protocol | To confirm |
| Whether side camera is left or right | Site survey required |
| Camera trigger and synchronization method | To confirm |
| Required cap-inspection coverage | To confirm |
| Count tolerance and automatic-approval policy | To validate |
| Minimum acceptable image quality | To define |
| Evidence retention period | To confirm |
| Report fields and export requirements | To confirm |
| Weighbridge / RFID / ERP integration | Optional; requirements unknown |
| Future unloading scan-line workflow | Separate phase |

------------------------------------------------------------------------

## 20. Project Status

This README documents the proposed system design and implementation
direction. It does not claim that all listed modules, AI models, camera
integrations, or automatic-approval capabilities are already
implemented.

As development progresses, update this section with completed modules,
model versions and evaluation results, supported camera configurations,
tested hardware, known limitations, and deployment instructions.

------------------------------------------------------------------------

## 21. Reference

This README uses headings and technical concepts from the supplied
reference document, **"AI Based Cylinder Counting & Inspection System"
(ComSys IT)**.

The reference describes offline cylinder counting, safety-cap detection,
camera evidence, reporting, Django-based application architecture,
PostgreSQL with optional pgvector, and a scan-based unloading workflow.
The three-camera loaded-truck inspection and exception-based approval
workflow here reflect the project direction discussed for this
implementation.

------------------------------------------------------------------------

**Built with Python, Django, Computer Vision, and Industrial Camera
Systems.**
