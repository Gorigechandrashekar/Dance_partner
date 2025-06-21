import cv2 as cv
import mediapipe as mp
import time
import numpy as np


class PoseDetectori:
    def __init__(self, mode=False, smooth=True, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img,img1, draw=True):
        imgRGB = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            landmark_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=0)
            connection_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255,0,0), thickness=10)
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,landmark_spec,connection_spec)
        return img

    def findPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append((cx, cy))
                # if draw:
                #     cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return lmlist


    def create_pose_mask(img, landmarks, radius=30):
        mask = np.zeros(img[:2], dtype=np.uint8)
        for (x, y) in landmarks:
            cv.circle(mask, (x, y), radius, 255, -1)
        return mask











# def main():
#     cap = cv.VideoCapture(0)
#     # detector = PoseDetector()
#     picture = None
#     ptime = 0

#     while True:
#         success, img = cap.read()
#         if not success:
#             break

#         img = cv.flip(img, 1) 

#         img = detector.findPose(img, draw=True)
#         cv.imshow("position",img)
#         lmlist = detector.findPosition(img, draw=True)
#         cTime = time.time()
#         fps = 1 / (cTime - ptime) if cTime != ptime else 0
#         ptime = cTime

#         cv.putText(img, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2)
#         cv.imshow("Invisible Pose", img)

#         # Exit condition
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv.destroyAllWindows()


# if __name__ == "__main__":
#     main()



    

        
    
