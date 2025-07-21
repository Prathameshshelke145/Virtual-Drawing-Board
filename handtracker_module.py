import cv2 as cv
import mediapipe as mp

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=mode, 
                                         max_num_hands=max_hands, 
                                         min_detection_confidence=detection_confidence, 
                                         min_tracking_confidence=tracking_confidence)

    def process_frame(self, frame, draw=True):
        """Processes the frame and returns hand landmarks in pixel coordinates."""
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        landmarks_list = []

        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS)
                
                h, w, c = frame.shape
                hand_points = []
                for id, lm in enumerate(hand_landmark.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)  # Convert to pixel coordinates
                    hand_points.append((id, cx, cy))
                
                landmarks_list.append(hand_points)

        return frame, landmarks_list
