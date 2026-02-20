import cv2
import onnxruntime as ort
import numpy as np
import os
import time
import csv
import gps
from datetime import datetime

# ================= CONFIG =================
MODEL_PATH = "model/best.onnx"
VIDEO_PATH = "videos/test.mp4"

IMG_SIZE = 320
CONF_THRES = 0.6
NMS_IOU = 0.25

FRAME_SKIP = 2
SNAP_INTERVAL = 15
FPS_UPDATE_INTERVAL = 10
SHOW_WINDOW = True
# ==========================================

os.makedirs("outputs/snapshots", exist_ok=True)

# -------- GPS Setup --------
gps_session = gps.gps(mode=gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)

def get_gps():
    try:
        report = gps_session.next()
        if report['class'] == 'TPV':
            latitude = getattr(report, 'lat', None)
            longitude = getattr(report, 'lon', None)

            if latitude is not None and longitude is not None:
                return round(latitude, 6), round(longitude, 6)
    except:
        pass

    return None, None


# -------- Load ONNX model --------
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name


# -------- Video --------
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Cannot open video file"


# -------- CSV Logging --------
log_path = "outputs/anomaly_log.csv"
log_file = open(log_path, "w", newline="")
writer = csv.writer(log_file)

writer.writerow([
    "timestamp",
    "class",
    "confidence",
    "bbox",
    "latitude",
    "longitude"
])

log_file.flush()


# -------- Runtime vars --------
frame_id = 0
last_snap = -100
fps = 0.0
t0 = time.time()
frame_counter = 0

print("Raspberry Pi inference started with REAL GPS logging")


# -------- Main Loop --------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    if frame_id % FRAME_SKIP != 0:
        continue

    frame_counter += 1

    # -------- Preprocess --------
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    preds = session.run(None, {input_name: img})[0][0]

    boxes = []
    scores = []
    classes = []

    # -------- Parse YOLO Output --------
    h0, w0 = frame.shape[:2]
    sx, sy = w0 / IMG_SIZE, h0 / IMG_SIZE

    for det in preds:

        obj_conf = det[4]
        cls_scores = det[5:]
        cls_id = int(np.argmax(cls_scores))
        score = float(obj_conf * cls_scores[cls_id])

        if score < CONF_THRES:
            continue

        cx, cy, w, h = det[:4]

        x = int((cx - w/2) * sx)
        y = int((cy - h/2) * sy)
        bw = int(w * sx)
        bh = int(h * sy)

        boxes.append([x, y, bw, bh])
        scores.append(score)
        classes.append(cls_id)


    # -------- Apply NMS --------
    keep = []

    if boxes:
        keep = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, NMS_IOU)


    # -------- Draw & Log --------
    if len(keep) > 0:

        for k in keep.flatten():

            x, y, w, h = boxes[k]
            score = scores[k]

            label = "pothole" if classes[k] == 1 else "obstacle"

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )


            # -------- Snapshot + GPS Logging --------
            if frame_id - last_snap >= SNAP_INTERVAL:

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save snapshot
                cv2.imwrite(
                    f"outputs/snapshots/{label}_{timestamp}.jpg",
                    frame
                )

                # Get real GPS coordinates
                latitude, longitude = get_gps()

                # Write CSV log
                writer.writerow([
                    timestamp,
                    label,
                    round(score, 3),
                    [x, y, x+w, y+h],
                    latitude,
                    longitude
                ])

                log_file.flush()

                last_snap = frame_id


    # -------- FPS Calculation --------
    if frame_counter % FPS_UPDATE_INTERVAL == 0:

        fps = frame_counter / (time.time() - t0)


    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )


    # -------- Display --------
    if SHOW_WINDOW:

        cv2.imshow(
            "Road Anomaly Detection with GPS",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break


# -------- Cleanup --------
cap.release()

log_file.close()

cv2.destroyAllWindows()


print(f"Average FPS: {fps:.2f}")
print("CSV logging working")
print("Snapshots working")
print("DONE")