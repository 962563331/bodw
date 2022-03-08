
import cv2
import time
import mediapipe as mp
import os
import datetime
from tqdm import tqdm

# 导入solution
mp_pose = mp.solutions.pose


mp_drawing = mp.solutions.drawing_utils


pose = mp_pose.Pose(static_image_mode=False,
            #    model_complexity=1,
                    smooth_landmarks=True,
           #        enable_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


def process_frame(img):
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(img_RGB)

    # 可视化
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # look_img(img)

    # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    #     # BGR转RGB
    #     img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     results = hands.process(img_RGB)

    #     if results.multi_hand_landmarks: # 如果有检测到手
    #
    #         for hand_idx in range(len(results.multi_hand_landmarks)):
    #             hand_21 = results.multi_hand_landmarks[hand_idx]
    #             mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS)

    return img

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 打开cap
cap.open(0)

cnt = 1

# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()

    cv2.namedWindow('my_window', 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
    cv2.resizeWindow('my_window', 1200, 700)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # text = 'Width: ' + str(cap.get(3)) + ' Height:' + str(cap.get(4))
    datet = str(datetime.datetime.now())
    # frame = cv2.putText(frame, text, (10, 50), font, 1,
    #                     (0, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, datet, (10, 100), font, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)

    if not success:
        print('Error')
        break

    ## !!!处理帧函数
    frame = process_frame(frame)

    dir = 'D:/frames'
    frame_name = os.path.join(dir, ('frame_' + str(cnt) + '.jpg'))

    cv2.imwrite(frame_name, frame)
    cnt = cnt + 1

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:
        break


cap.release()


cv2.destroyAllWindows()
