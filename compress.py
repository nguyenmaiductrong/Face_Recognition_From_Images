import numpy as np
import os
import transform

output_path = "embeddings"
def compress_faces(directory_path, option):
    '''
        Nén các embeddings lại thành 1 file .npz
        option = "face" luu output vao thu muc embeddings/face
        option = "portrait" luu output vao thu muc embeddings/portrait
    '''

    if option == "face":
        output_path = "embeddings/face"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        filename = os.path.basename(directory_path) + ".npz"
        filepath = os.path.join(output_path, filename)   
    elif option == "portrait":
        output_path = "embeddings/portrait"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        filename = "input" + ".npz"
        filepath = os.path.join(output_path, filename)       
    embeddings_dict = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory_path, filename)
            embedding = transform.transform(img_path)
            embeddings_dict[filename] = embedding
    np.savez_compressed(filepath, **embeddings_dict)
