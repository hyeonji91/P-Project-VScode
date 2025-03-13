
import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import itertools


num_of_video = 3000
video_root_path = "C:\\Users\\HyeonjiKim\\Documents\\GitHub\\signLanguageDataset\\0001~3000(video)"

video_file_list = os.listdir(video_root_path) # video 이름
video_path_list = np.array([os.path.join(video_root_path, file) for file in video_file_list]) # video file path

# label 읽기
df = pd.read_excel('C:/Users/HyeonjiKim/Documents/GitHub/signLanguageDataset/KETI-2017-SL-Annotation-v2_1.xlsx') # 엑셀에서 목록 불러오기
df.sort_values(by = '번호', ascending=True, inplace=True) # 파일 이름(번호)순으로 정렬
df['번호'] = "KETI_SL_"+df['번호'].astype(str).str.zfill(10) +".avi" # 파일 이름 형식인 KETI_SL_0000000000.avi 로 생성성
label_dic = dict(zip(df['번호'], df['한국어'])) # 딕셔너리 {파일이름 : label} 
# label_dic1 = dict(itertools.islice(label_dic.items(), 3000))
# print(label_dic1)


for file, label in label_dic1.items():
    cap = cv2.VideoCapture(os.path.join(video_root_path, file)) #비디오 읽을 path setting
    
    # 비디오의 fps, 너비 높이 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = min(width, height)

    # 비디오 저장 설정 (코덱: mp4v)
    video_save_path = "C:/Users/HyeonjiKim/Documents/GitHub/signLanguageDataset/0001~3000(video_crop)"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(video_save_path, file), fourcc, fps, (size, size))


    while cap.isOpened():
        success, frame = cap.read()
        if not success: # 비디오 못 읽었으면 넘어가기
            break

        # 중앙을 기준으로 정사각형 크롭
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        cropped_frame = frame[start_y:start_y+size, start_x:start_x+size]

        # 비디오 저장
        out.write(cropped_frame)
    

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()

print("비디오 크롭 완료! 🎥✅")