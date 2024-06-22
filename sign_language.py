import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)
            
           
            for tip_id in finger_tips:
                x = int(lm_list[tip_id].x * w)
                y = int(lm_list[tip_id].y * h)
                cv2.circle(img, (x, y), 10, (255, 0, 0), cv2.FILLED)
            
            
            finger_fold_status = []
            for tip_id in finger_tips:
                if lm_list[tip_id].x < lm_list[tip_id - 2].x:
                    cv2.circle(img, (int(lm_list[tip_id].x * w), int(lm_list[tip_id].y * h)), 10, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)
            
            
            if all(finger_fold_status):
                thumb_up = lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y
                if thumb_up:
                    cv2.putText(img, 'LIKE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, 'DISLIKE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS, 
                mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,255,0), thickness=4, circle_radius=2))

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
