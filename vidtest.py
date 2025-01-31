import cv2

def main():
    #gst_pipeline = "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink"
    gst_pipeline = "filesrc location=cars.mp4 ! decodebin ! videoconvert ! appsink"

    #cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture("cars.mp4")

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Traffic Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
