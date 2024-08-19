import cv2
import torch
import os
from facenet_pytorch import MTCNN

mtcnn = MTCNN(image_size=160, margin=0)
def face_detect(input_path, option):
    '''
        Cắt khuôn mặt của từng người trong 1 tấm ảnh rồi lưu vào 1 thư mục.
        Nếu là ảnh cần tìm kiếm thì lưu kết quả trong thư mục portrait.
        Nếu là ảnh trong kho ảnh cần tìm kiếm thì lưu kết quả trong thư mục face.
    '''
    if option == "portrait":  
        output_dir = "process/portrait"
    elif option == "face":
        output_dir = "process/face"
    name_dir = os.path.basename(input_path)
    name_dir, _ = os.path.splitext(name_dir)
    output_path = os.path.join(output_dir, name_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img = cv2.imread(input_path)
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            face_img = img[y1:y2, x1:x2]
            if face_img is not None and face_img.size != 0:  
                face_img_resized = cv2.resize(face_img, (160, 160))
                output_path1 = os.path.join(output_path, f"{i}.jpg")
                cv2.imwrite(output_path1, face_img_resized)