import cv2
import mediapipe as mp
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from st_gcn import STGCN


# 키포인트 추출
def extract_keypoints(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        keypoints = []

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic()
        mp_draw = mp.solutions.drawing_utils
        mp_draw_styles = mp.solutions.drawing_styles


        # 이미지 입력 캡처 및 처리
        # media pipe 는 RGB
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            frame_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #mediapipe는 RGB를 입력으로 받음
            results = holistic.process(frame_RGB) # 키포인트 추출

            # 몸, 손 키포인트 추출 
            if results.pose_landmarks:
                pose = np.array([[kp.x, kp.y, kp.z] for kp in results.pose_landmarks.landmark])
            else:
                pose = np.zeros((33,3)) # 33개 keypoints
            if results.left_hand_landmarks:
                left_hand = np.array([[kp.x, kp.y, kp.z] for kp in results.left_hand_landmarks.landmark])
            else:
                left_hand = np.zeros((21,3))
            if results.right_hand_landmarks:
                right_hand = np.array([[kp.x, kp.y, kp.z] for kp in results.right_hand_landmarks.landmark])
            else:
                right_hand = np.zeros((21,3))

            
            frame_keypoints = np.concatenate([pose, left_hand, right_hand])
            keypoints.append(frame_keypoints)

        cap.release()
        holistic.close()
        return np.array(keypoints)

    except Exception as e:
        print(f"예외 발생: {e}")
        return None

# 입력: 비디오가 들어있는 파일의 path
# def extract_video_list_keypoint(video_root_path):
#     video_path_list = os.listdir(video_root_path) # video 이름 
#     os.chdir(video_root_path) # 작업디렉토리 번경
    

#     for video_path in video_path_list[:1]:
#         print('start')
#         keypoints = extract_keypoints(video_path)
#         print('fin')
#         print(keypoints)
#         print(keypoints.shape)
    
#         return keypoints


def data_preprocessing(keypoints, num_person = 1, num_channels = 3):
    """
    ST-GCN 입력 형식으로 키포인트 데이터를 변환합니다.
    Args:
        keypoints: (num_frames, num_nodes, 3) 형식의 키포인트 데이터
        num_person: 프레임당 사람의 수 (기본값: 1)
        num_channels: 좌표 차원 수 (기본값: 3 - x, y, z)
    Returns:
        Tensor: (N, C, T, V, M) 형식의 데이터
        N : batch_size
        C : keypoint의 차원
        T : fps
        V : 한 프레임당 keypoint 개수
        M : 사람 수
    """
    num_frames, num_nodes, _ = keypoints.shape

    data = np.zeros((1, num_channels, num_frames, num_nodes, num_person))
    for t in range(num_frames):
        data[0, :, t, :, 0] = keypoints[t].T

    return torch.tensor(data, dtype=torch.float)



class SignLangDataSet(Dataset):
    def __init__(self, video_paths, labels):
        """
        Args:
            video_paths (list): 영상 경로 리스트
            labels (list): 각 영상에 대한 레이블 리스트
        """
        self.video_paths = video_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        keypoints = extract_keypoints(video_path) # (T, V, C) 형식 반환
        data = data_preprocessing(keypoints) # (N, C, T, V, M) 형식으로 변환

        return data, torch.tensor(label, dtype=torch.long)


# 훈련
def train(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.
    train_progress = 0

    for batch_idx, (data, label) in enumerate(dataloader):
        data, label = data.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_progress += len(data)
        
        print("Train epoch : {} [{}/{}], learning cost {}, avg cost {}".format(
            epoch, train_progress, len(dataloader.dataset),
            loss.item(),
            total_loss / (batch_idx + 1)
        ))
        
    return total_loss

def evaluate(model, dataloader, criterion):
    model.eval()
    eval_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            data, label = data.to(DEVICE), label.to(DEVICE)

            output = model(data)
            eval_loss += criterion(output, label)
            prediction = torch.argmax(output, 1)
            correct += (output == prediction).sum().item()
    
    eval_loss /= len(dataloader.dataset)
    eval_accuracy = 100 * correct / len(dataloader.dataset)
    return eval_loss, eval_accuracy



### GPU Setting ###
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else 'cpu')
print(DEVICE)


num_of_video = 3000
video_root_path = "F:/HyeonjiKim/Downloads/signLanguageDataset/0001~3000(video)"
# keypoints = extract_video_list_keypoint(video_root)
# data_preprocessing(keypoints)

# video file path 읽기
video_file_list = os.listdir(video_root_path) # video 이름
video_path_list = np.array([os.path.join(video_root_path, file) for file in video_file_list ])

# label읽기 
df = pd.read_excel('F:/HyeonjiKim/Downloads/signLanguageDataset/KETI-2017-SL-Annotation-v2_1.xlsx')
df.sort_values(by = '번호', ascending=True, inplace=True)
label_list = df['한국어'].tolist()
label_list = np.array(label_list[:num_of_video])

#데이터 생성
X_train, X_test, y_train, y_test = train_test_split(video_path_list, label_list, test_size = 0.2, random_state=42, stratify=label_list)
train_dataset = SignLangDataSet(X_train, y_train)
test_dataset = SignLangDataSet(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 모델 초기화
graph_args = {"layout": "openpose", "strategy": "spatial"}
model = STGCN(in_channels=3, num_class=2, graph_args=graph_args, edge_importance_weighting=True).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
epochs = 1

# 훈련 실행
for epoch in range(epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion, epoch)
    val_loss, val_accuracy = evaluate(model, train_dataloader, criterion)

    if val_accuracy > best:
        best = val_accuracy
        torch.save(model.state_dict(), "model/best_model.pth")
    print(f'[{epoch}] Validation Loss : {val_loss:.4f}, Accuracy : {val_accuracy:.4f}%')

print("[FINISH]")

