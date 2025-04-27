from ultralytics import YOLO
import cv2
import numpy as np
import sys
import time

class BaseYoloFinder():
    def prepare_self(self):
        pass

    def magic(self, frame):
        return frame

class PlainYoloFinder(BaseYoloFinder):
    def prepare_self(self):
        pass
    def magic(self, frame):
        results = model.predict(frame, classes=0, verbose=False, stream=True)
        for result in results:
            df = result.to_df()
            if 'box' in df.columns and 'confidence' in df.columns:
                boxes = (df[df['confidence'] > 0.4])['box'].to_list()
                for box in boxes:
                    xA, yA, xB, yB = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
                    
                    cv2.rectangle(frame, (xA, yA), (xB, yB), color=(0, 255, 0), thickness=2)
                    cv2.rectangle(frame, (xA, yA), (xB, yA-14), color=(0, 255, 0), thickness=-1)
                    cv2.putText(frame,f'x={int((xA+xB)/2)};y={int((yA+yB)/2)}', (xA, yA),cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(255, 255, 255))
        return frame

class OptimalYoloFinder(BaseYoloFinder):
    x_l, y_l, x_r, y_r = None, None, None, None
    counter = 0
    def prepare_self(self):
        self.x_l = None
        self.y_l = None
        self.x_r = None
        self.y_r = None
        self.counter = 0

    def magic(self, frame):
        cur_frame = None
        results = None

        if self.x_l is not None and self.y_l is not None:
            cur_frame = frame[self.y_l:self.y_r, self.x_l:self.x_r]
            cv2.rectangle(frame, (self.x_l,self.y_l),(self.x_r,self.y_r), color=(0, 0, 255), thickness=2)
        else:
            cur_frame = frame
        
        if cur_frame.shape[0] * cur_frame.shape[1] == 0:
            cur_frame = frame

        results = model.predict(cur_frame, classes=0, verbose=False, stream=True)

        max_x, max_y = 0, 0
        min_x, min_y = 1000000, 1000000
        need_upd = False
        for result in results:
            df = result.to_df()
            if 'box' in df.columns and 'confidence' in df.columns:
                boxes = (df[df['confidence'] > 0.4])['box'].to_list()
                need_upd = (len(boxes) > 0)
                for box in boxes:
                    xA, yA, xB, yB = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

                    min_x = min(min_x, xA, xB)
                    max_x = max(max_x, xA, xB)
                    min_y = min(min_y, yA, yB)
                    max_y = max(max_y, yA, yB)

                    cv2.rectangle(cur_frame, (xA, yA), (xB, yB), color=(0, 255, 0), thickness=2)
                    # cv2.rectangle(cur_frame, (xA, yA), (xB, yA-14), color=(0, 255, 0), thickness=-1)
                    # cv2.putText(cur_frame,f'({xA};{yA})<>({xB};{yB})', (xA, yA),cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(255, 255, 255))
        if need_upd:
            if self.x_l is not None:
                min_x += self.x_l
                min_y += self.y_l
                max_x += self.x_l
                max_y += self.y_l

            precent = 0.09
            def constraint(val, min, max):
                if val > max:
                    return max
                elif val < min:
                    return min
                else:
                    return val
                
            self.x_l = int(constraint(min_x * (1-precent), 0, frame.shape[1]))
            self.y_l = int(constraint(min_y * (1-precent), 0, frame.shape[0]))
            self.x_r = int(constraint(max_x * (1+precent), 0, frame.shape[1]))
            self.y_r = int(constraint(max_y * (1+precent), 0, frame.shape[0]))
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter > 5:
            self.prepare_self()
        
        return frame

def make_magic(ifile, ofile, executer: BaseYoloFinder):
    vid = cv2.VideoCapture(ifile)
    resolution = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    output = cv2.VideoWriter(ofile,fourcc,vid.get(cv2.CAP_PROP_FPS),resolution)

    executer.prepare_self()
    while(True):
        ret, frame = vid.read()
        if frame is None:
            print('Frame is none')
            break
        output.write(executer.magic(frame))

    vid.release()
    output.release()

    

if len(sys.argv) != 4:
    print('Input params: yolo model file, input file, output file')
    exit()

model = YOLO(sys.argv[1])
start_time = time.time()
make_magic(sys.argv[2], sys.argv[3], OptimalYoloFinder())
end_time = time.time()
print(f"Magic took {(end_time - start_time):.2f} seconds to complete.")