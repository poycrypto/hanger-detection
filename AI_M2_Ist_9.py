import cv2, math
import numpy as np
from ultralytics import YOLO
import snap7
import os, sys
import multiprocessing
import time
import keyboard
import PLC_Comm

def resource_path(relative):

    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative)

def put_boxes(box, frame, label):
    # bounding box
    global x1,y1,x2,y2
    confidence = math.ceil(box.conf[0]*100)/100

    #put text and box in cam
    color = (0,0,255) if label == 'nok' else (0,255,0)
    org = [x1, y1-5]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    cv2.putText(frame, ("%.2f") % confidence, org, font, fontScale, color, thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


def put_text(frame, text):
    org = [100, 100]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 5
    color = (0, 255, 0)
    thickness = 2
    cv2.putText(frame, text, org, font, fontScale, color, thickness)

def put_info(frame):
    if plc_connected:
        text = "PLC Baglanti: OK"
    else:
        text = "PLC Baglanti: NOK"

    x,y,w,h = 0,0,340,75

    # Draw black background rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,0), -1)
    # Add text
    cv2.putText(frame, text, (x + int(w/10), y + int(h/2)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def get_video(source):
    try:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    except:
        pass
    return cap

def recipe_conversion(program_no):

    if program_no == 0:
        recipe_no = 0

    elif program_no == 10 or program_no == 13 or program_no == 15:
        recipe_no = 1 # Tek sol

    elif program_no == 7 or program_no == 12 or program_no == 14:
        recipe_no = 2 # Tek sag
    
    elif program_no == 9:
        recipe_no = 3 # Cıft sol

    elif program_no == 8:
        recipe_no = 4 # Cıft sag

    elif program_no == 2 or program_no == 6 or program_no == 16 or program_no == 17:
        recipe_no = 5  #Tek askisiz

    elif program_no == 3 or program_no == 4 or program_no == 11:
        recipe_no = 6 # Cift askisiz
    
    return recipe_no

def main():
    global x1,y1,x2,y2, plc_connected
    ip = "10.20.17.229"
    plc_connected = False
    test_start = False
    recipe_no = 0
    plc = snap7.client.Client()

    cap = get_video(0)
    pt = resource_path('best_nok.pt')
    model = YOLO(pt)
    
    # pt2 =resource_path("handle_cover_small_v2_openvino_model/")
    # openvino_model = YOLO(pt2)

    while True:

        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (1728,972))
            print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
             
        else:
            cap.release()
            time.sleep(1)
            cap = get_video(0)
                        
        if plc.get_cpu_state() == 'S7CpuStatusUnknown':
            plc_connected = False
            try:
                plc.disconnect()
                time.sleep(1)
                plc.connect(ip,0,1)
            except:
                pass
        
        if plc.get_cpu_state() == 'S7CpuStatusRun':
            plc_connected = True
            test_start = PLC_Comm.read_bool(plc,20,2,0)
            
        if success == True:

            if plc_connected and test_start:
                recipe_no = recipe_conversion(PLC_Comm.read_int(plc,20,0))
                
                results = model.predict(frame, stream=True, iou = 0.1, conf = 0.70, imgsz=640)
                #results = openvino_model(frame, imgsz = 640, iou = 0.1)
                test_ok = False
                labels = []
                for r in results:
                    boxes = r.boxes
                    boxes = sorted(boxes, key=lambda x: x.xyxy[0][0])
                    
                    for box in boxes:
                        label = r.names[int(box.cls[0])]
                        labels.append(label) 
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                        put_boxes(box, frame, label) 
                
                if recipe_no == 1 and len(boxes) == 2:
                    if labels[0] == "bezel" and labels[1] == "askilik":
                        test_ok = True

                elif recipe_no == 2 and len(boxes) == 2:
                    if labels[0] == "askilik" and labels[1] == "bezel":
                        test_ok = True
            
                elif recipe_no == 3 and len(boxes) == 4:
                    if labels[0] == "bezel" and labels[1] == "askilik" and labels[2] == "bezel" and labels[3] == "askilik":
                        test_ok = True

                elif recipe_no == 4 and len(boxes) == 4:
                    if labels[0] == "askilik" and labels[1] == "bezel" and labels[2] == "askilik" and labels[3] == "bezel":
                        test_ok = True
                
                elif recipe_no == 5 and len(boxes) == 2:
                    if labels[0] == labels[1] == "bezel":
                        test_ok = True

                elif recipe_no == 6 and len(boxes) == 4:
                    if labels[0] == labels[1] == labels[2] == labels[3] == "bezel":
                        test_ok = True        

                if test_ok:
                    PLC_Comm.write_bool(plc,20,10,0,test_ok)
                    print('OK')

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        put_info(frame)
        cv2.imshow('Bezel Askilik Kontrol', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()