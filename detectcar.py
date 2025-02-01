import cv2
import numpy as np
from collections import OrderedDict

# Inisialisasi model YOLO
net = cv2.dnn.readNet("yolov7-tiny.weights", "yolov7-tiny.cfg")
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Garis penghitungan
ENTRY_LINE_Y = 300
EXIT_LINE_Y = 350

class TrackedObject:
    def __init__(self, centroid, box):
        self.centroid = centroid
        self.box = box
        self.crossed_entry = False
        self.counted = False

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=50):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box):
        self.objects[self.next_id] = TrackedObject(centroid, box)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([(x + w//2, y + h//2) for (x, y, w, h) in detections])
        input_boxes = np.array(detections)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_boxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[obj_id].centroid for obj_id in object_ids])

            distances = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                if distances[row, col] < self.max_distance:
                    self.objects[object_id].centroid = input_centroids[col]
                    self.objects[object_id].box = input_boxes[col]
                    self.disappeared[object_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)

            unused_object_ids = set(object_ids) - set(object_ids[i] for i in rows)
            for object_id in unused_object_ids:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_input_ids = set(range(len(input_centroids))) - used_cols
            for i in unused_input_ids:
                self.register(input_centroids[i], input_boxes[i])

        return self.objects

# Inisialisasi tracker
tracker = CentroidTracker()
vehicle_count = 0

def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["car", "bus", "truck", "motorcycle"]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w//2)
                y = int(center_y - h//2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [boxes[i] for i in indexes.flatten()] if len(indexes) > 0 else []

def process_frame(frame):
    global vehicle_count
    detections = detect_objects(frame)
    objects = tracker.update(detections)

    # Update status dan gambar bounding box
    for object_id, tracked_obj in objects.items():
        x, y, w, h = tracked_obj.box
        centroid = tracked_obj.centroid
        
        # Update status garis masuk
        if not tracked_obj.crossed_entry and centroid[1] > ENTRY_LINE_Y:
            tracked_obj.crossed_entry = True
        
        # Update status garis keluar
        if tracked_obj.crossed_entry and not tracked_obj.counted and centroid[1] > EXIT_LINE_Y:
            vehicle_count += 1
            tracked_obj.counted = True
        
        # Gambar bounding box jika belum terhitung
        if not tracked_obj.counted:
            color = (0, 255, 0) if tracked_obj.crossed_entry else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID {object_id}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def main():
    #gst_pipeline = "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink"
    gst_pipeline = "filesrc location=cars.mp4 ! decodebin ! videoconvert ! appsink"

    #cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture("cars.mp4")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = process_frame(frame)
        
        # Gambar garis penghitungan
        cv2.line(frame, (0, ENTRY_LINE_Y), (frame.shape[1], ENTRY_LINE_Y), (0, 255, 255), 2)
        cv2.line(frame, (0, EXIT_LINE_Y), (frame.shape[1], EXIT_LINE_Y), (0, 0, 255), 2)
        cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Traffic Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
