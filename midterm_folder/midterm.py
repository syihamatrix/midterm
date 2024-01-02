import cv2
from ultralytics import YOLO



# Load a model(using yolo)
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('best.pt')  # load an official model  # best.pt ambik dari google colab lepas dah download bahagian custom trained tekan file run-detects-wights-best


#vid = cv2.VideoCapture(0)   # webcam can be call cv2.VideoCaptue(0) # no webcam (0) # Kalau nak panggil video buad ('hahahaha'.mp4)
vid = cv2.VideoCapture(1)   
while True:
    ret, frame = vid.read()

    results=model(frame,stream=True)  # Feedforward

    for r in results:  #(Doing loop)
        boxes=r.boxes
        for bbox in boxes:
            x1,y1,x2,y2=bbox.xyxy[0]      #xyxy mean topleft and bottomm right coordinate
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)    # make all integer or float


            cls_idx=int(bbox.cls[0])      # Cara nak bagi detect benda apa(coding ygni dgn clas_name)
            cls_name=model.names[cls_idx]

            conf=round(float(bbox.conf[0]),2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4) # Buad bentuk (rectangle shape) dan (warna=255) apa (line dan thickness=4)
            cv2.putText(frame,f'{cls_name} {conf}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow('My Vid',frame)

    cv2.waitKey(1)    # 0 utk tekan , 1 utk stay video

vid.release()
cv2.destroyAllWindows()    # Nak tutup webcam tekan ctrl + C