import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import json

num_of_video = 3000
video_root_path = "C:/Users/HyeonjiKim/Documents/GitHub/signLanguageDataset/0001~3000(video)"

video_file_list = os.listdir(video_root_path) # video 이름
video_path_list = np.array([os.path.join(video_root_path, file) for file in video_file_list]) # video file path

# label 읽기
df = pd.read_excel('C:/Users/HyeonjiKim/Documents/GitHub/signLanguageDataset/KETI-2017-SL-Annotation-v2_1.xlsx') # 엑셀에서 목록 불러오기
df.sort_values(by = '번호', ascending=True, inplace=True) # 파일 이름(번호)순으로 정렬
df['번호'] = "KETI_SL_"+df['번호'].astype(str).str.zfill(10) +".avi" # 파일 이름 형식인 KETI_SL_0000000000.avi 로 생성성
label_dic = dict(zip(df['번호'], df['한국어'])) # 딕셔너리 {파일이름 : label} 

# 키포인트 추출
keypoint_list = [] # [[keypoint], 라벨, file]
for file, label in label_dic.items():
    cap = cv2.VideoCapture(os.path.join(video_root_path, file)) #비디오 읽을 path setting
    
    # 비디오의 fps, 너비 높이 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = min(width, height)

    keypoints = []
    print("#### 시작 ####")

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

        # 몸 키포인트 추출
        if results.pose_landmarks:
            pose = np.array([[kp.x, kp.y, kp.z] for kp in results.pose_landmarks.landmark])
        else:
            pose = np.zeros((33,3))  # 33개 keypoints
        # 왼손 키포인트 추출
        if results.left_hand_landmarks:
            left_hand = np.array([[kp.x, kp.y, kp.z] for kp in results.left_hand_landmarks.landmark])
        else:
            left_hand = np.zeros((21,3))
        # 오른손 키포인트 추출
        if results.right_hand_landmarks:
            right_hand = np.array([[kp.x, kp.y, kp.z] for kp in results.right_hand_landmarks.landmark])
        else:
            right_hand = np.array((21,3))
        
        frame_keypoints = np.concatenate([pose, left_hand, right_hand])
        keypoints.append(frame_keypoints)
    
    # list로 저장
    keypoint_list.append([keypoints, label, file])
    cap.release()
    holistic.close()
    
print("##### 키포인트 변환 끝 #####")
# json타입으로 저장하기 
save_path = 'data/keypoint1~3000.json'
with open("data.json", "w") as f:
    json.dump(keypoint_list, f)

print("##### 저장 끝 #####")
# 불러오기 
with open(save_path, 'r') as f:
    loaded_list = json.load(f)
print(loaded_list)