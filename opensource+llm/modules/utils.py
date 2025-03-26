import os
import cv2
import numpy as np

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# vector normalization
def Vector_Normalization(joint):
    # Compute angles between joints
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # Child joint
    v = v2 - v1 
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

    angle = np.degrees(angle) # Convert radian to degree

    angle_label = np.array([angle], dtype=np.float32)

    return v, angle_label


def Merge_Jamo_With_LLM(zamo_str, processor, model):
    messages = [
    {'role': 'user','content': [
        {'type': 'text','text': f"다음 자음과 모음을 조합하여 올바른 한글 단어로 변환해줘: {zamo_str}"}
        ]},
    ]

    input_text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

    inputs = processor(
        images=None,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs,max_new_tokens=256,temperature=0.1,eos_token_id=processor.tokenizer.convert_tokens_to_ids('<|eot_id|>'),use_cache=False)

    return processor.decode(output[0]).strip()

