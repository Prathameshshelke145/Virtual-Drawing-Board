import numpy as np
import cv2 as cv
from handtracker_module import HandTracker

tracker = HandTracker(detection_confidence=0.85)
xp, yp = 0, 0
id = 0
inputFolder = r"D:\Users\ASUS\OneDrive\Desktop\Untitled design"
header = "1.jpg"
cap = cv.VideoCapture(0)

# Initialize the drawing frame
drawing_frame = None

def counting(frame, Landmark_list):
    finger = []
    values = [4, 8, 12, 16, 20]

    if Landmark_list:
        if len(Landmark_list[0]) >= max(values):  # Ensure we have enough landmarks
            for i in range(1, 5):  # Iterate over finger tips
                if Landmark_list[0][values[i]][2] < Landmark_list[0][values[i] - 2][2]:
                    finger.append(1)
                else:
                    finger.append(0)

            count = finger.count(1)
            return count
    return 0


# Function to select the header based on x1 position
def header_selection(x1, count):
    if x1 > 243 and x1 < 335:
        color = (0, 0, 255)
        return "1.jpg", 1, color
    elif x1 > 440 and x1 < 530:
        color = (255, 0, 0)
        return "2.jpg", 2, color
    elif x1 > 677 and x1 < 767:
        color = (0, 255, 0)
        return "3.jpg", 3, color
    elif x1 > 838 and x1 < 928:
        color = (0, 255, 255)  # Fixed typo here (from 0.255 to 0, 255, 255)
        return "4.jpg", 4, color
    elif x1 > 1002 and x1 < 1080:
        color = (0, 0, 0)
        return "5.jpg", 5, color
    return "5.jpg", 5, (0, 0, 0)  # Default to the first header


def action(frame, color):
    global xp, yp, drawing_frame
    if Landmark_list:
        if len(Landmark_list[0]) > 8:
            x1, y1 = Landmark_list[0][8][1], Landmark_list[0][8][2]
            if drawing_frame is None:
                # Create a blank drawing frame if not already created
                drawing_frame = np.zeros_like(frame)
            # Draw on the drawing frame
            if (xp , yp) == (0 , 0):
                  (xp , yp) = (x1,y1)
            cv.line(drawing_frame, (xp, yp), (int(x1), int(y1)), color, 3)
            xp, yp = int(x1), int(y1)
    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    frame = cv.resize(frame, (1080, frame.shape[0] + 200))  # (width, height)
    frame = cv.flip(frame, 1)

    # Process frame using hand tracker
    frame, Landmark_list = tracker.process_frame(frame)
    count = counting(frame, Landmark_list)

    if count == 2:  # Selection mode
        # print("Selection mode")
        if Landmark_list:
            if len(Landmark_list[0]) > 4:
                x1, y1 = Landmark_list[0][4][1], Landmark_list[0][4][2]
                print(f"x1: {x1}, y1: {y1}")
                if y1 < 275:  # Condition to check if the hand is in the top region
                    header, id, color = header_selection(x1, count)  # Update the header based on x1 position
                    print(f"Selected header: {header}")

    if count == 1:  # Drawing mode
        # print("Drawing mode")
        if Landmark_list:
            if len(Landmark_list[0]) > 8:
                x1, y1 = Landmark_list[0][8][1], Landmark_list[0][8][2]
                cv.circle(frame, (int(x1), int(y1)), 15, color, -1)  # Draw circle with updated color
                frame = action(frame, color)

    # If there is a drawing, merge it with the current frame
    if drawing_frame is not None:
        frame = cv.add(frame, drawing_frame)

    # Load and place the selected header image
    head = cv.imread(header)
    if head is None:
        print(f"Error: Could not load header image from {header}")
        break  # Exit loop if the header image is missing

    head = cv.resize(head, (1080, 100))  # Resize header to match width=1080, height=100

    # Ensure the frame is tall enough to fit the header
    if frame.shape[0] >= 100 and frame.shape[1] == 1080:
        frame[0:100, 0:1080] = head  # Place header at the top
    else:
        print("Error: Mismatch in frame and header dimensions.")

    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
