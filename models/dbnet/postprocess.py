import numpy as np
import cv2

class SegDetectorRepresenter:
    def __init__(self, thresh=0.3, box_thresh=0.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
    
    def __call__(self, preds, images):
        boxes_all = []
        for pred, img in zip(preds, images):
            prob_map = pred[0]  # [H, W]
            _, binary_map = cv2.threshold(prob_map, self.thresh, 255, cv2.THRESH_BINARY)
            binary_map = binary_map.astype(np.uint8)
            contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                box = cv2.minAreaRect(contour)
                box_points = cv2.boxPoints(box)
                box_points = np.int0(box_points).tolist()
                boxes.append(box_points)
            boxes_all.append(boxes)
        return boxes_all, None
