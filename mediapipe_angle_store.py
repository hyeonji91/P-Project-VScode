## mediapipe로 인식후 angle계산해서 저장


import os, time
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import json
import itertools
import logging
import warnings

# TensorFlow 경고 메시지 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# MediaPipe 경고 메시지 제거
logging.getLogger("mediapipe").setLevel(logging.ERROR)
# Python 자체 경고 메시지 제거
warnings.filterwarnings("ignore")


# input : mediapipe 손 keypoint (21,4)
# out : angle (15,)
def cal_angle(joint):
    # 벡터 계산
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
    v = v2 - v1 # [20, 3]

    # normalize v : 길이로 나누기
    v = v/np.linalg.norm(v, axis=1)[:, np.newaxis]

    # arccos dot product로 앵글 구하기
    angle = np.arccos(np.einsum('nt, nt->n',
                                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

    # radian을 degree(도)로 변경
    angle = np.degrees(angle) 

    angle = np.array([angle], dtype=np.float32)
    return angle

num_of_video = 3000
video_root_path = "C:/Users/HyeonjiKim/Documents/GitHub/signLanguageDataset/0001~3000(video_crop)"

video_file_list = os.listdir(video_root_path) # video 이름
video_path_list = np.array([os.path.join(video_root_path, file) for file in video_file_list]) # video file path

# 파일 읽기
df = pd.read_excel('C:/Users/HyeonjiKim/Documents/GitHub/signLanguageDataset/KETI-2017-SL-Annotation-v2_1.xlsx') # 엑셀에서 목록 불러오기
df.sort_values(by = '번호', ascending=True, inplace=True) # 파일 이름(번호)순으로 정렬
df['번호'] = "KETI_SL_"+df['번호'].astype(str).str.zfill(10) +".avi" # 파일 이름 형식인 KETI_SL_0000000000.avi 로 생성성
label_dic = dict(zip(df['번호'], df['한국어'])) # 딕셔너리 {파일이름 : label} 

label_dic1 = dict(itertools.islice(label_dic.items(), 3000))
print(label_dic1)

### label - idx mapping정보 가져오기
import pickle
with open('data/label_to_idx.pickle', 'rb') as f:
    label_to_idx = pickle.load(f)
print(label_to_idx)

# 키포인트 추출
data_list = [] # [left_kp, right_kp, left_angle, right_angle, 라벨]]
for file, label in label_dic1.items():
    print(label, " 시작")
    cap = cv2.VideoCapture(os.path.join(video_root_path, file)) #비디오 읽을 path setting

    data = []

    holistic = mp.solutions.holistic.Holistic()

    # 이미지 입력 캡쳐 및 처리
    # media pipe는 RGB를 입력받음음
    while cap.isOpened():
        success, frame = cap.read()
        if not success: # 비디오 못 읽었으면 넘어가기
            break

        # cv read결과는 BGR, mediapipe는 RGB를 입력으로 받음
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        # 키포인트 추출
        results = holistic.process(frame_RGB) 


        # 왼손 키포인트 추출
        if results.left_hand_landmarks:
            joint_left = np.zeros((21,4))
            # 키포인트 추출
            for j, lm in enumerate(results.left_hand_landmarks.landmark):
                joint_left[j] = [lm.x, lm.y, lm.z, lm.visibility]
            # 앵글 계산 [15,]
            angle_left = cal_angle(joint_left)
        else:
            joint_left = np.zeros((21,4))
            angle_left = [np.zeros((15,))]

        # 오른손 키포인트 추출
        if results.right_hand_landmarks:
            joint_right = np.zeros((21,4))
            # 키포인트 추출
            for j, lm in enumerate(results.right_hand_landmarks.landmark):
                joint_right[j] = [lm.x, lm.y, lm.z, lm.visibility]
            # 앵글 계산
            angle_right = cal_angle(joint_right)
        else:
            joint_right = np.zeros((21,4))
            angle_right = [np.zeros((15,))]
        
        frame_angle = np.concatenate([angle_left, angle_right])
        angle_label = np.append(frame_angle, label_to_idx[str(label)])

        d = np.concatenate([joint_left.flatten(), joint_right.flatten(), angle_label])
        data.append(d) # d = 21x4 + 21x4 + 15 + 15 + 1 = 199 [199,]
     
    # list로 저장, 개별 요소를 저장
    data_list.extend(data)
    cap.release()
    holistic.close()
    
print("##### 키포인트 변환 끝 #####")

created_time = int(time.time())
# npy로 저장 
save_raw_path = f'data/raw1~3000_{created_time}'
data_list = np.array([data_list])
print(data_list.shape)
np.save(save_raw_path, data_list)




