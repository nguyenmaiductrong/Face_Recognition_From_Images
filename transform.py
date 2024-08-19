import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(image_size=160, margin=0)
model = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face(image):
    face = mtcnn(image)
    return face

def get_embedding(model, face_tensor):
    face_tensor = torch.tensor(face_tensor)
    if len(face_tensor.shape) == 3:
        face_tensor = face_tensor.unsqueeze(0)
    face_embedding = model(face_tensor)
    return face_embedding.detach().numpy()[0]

def transform(face_path):
    img = cv2.imread(face_path)
    face = extract_face(img)
    if face is not None:
        embedding = get_embedding(model, face)
        return embedding
    return None