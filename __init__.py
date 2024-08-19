import os
import detect
import compress
import compare
import shutil

# Tiền xử lý ảnh trong kho
image_library_path = "library"

for filename in os.listdir(image_library_path):
    if filename.endswith('.jpg'):
        file_path = os.path.join(image_library_path, filename)
        detect.face_detect(file_path, "face")
face_path = "process/face"
for subdir in os.listdir(face_path):
    subdir_path = os.path.join(face_path, subdir)
    compress.compress_faces(subdir_path, "face")

# Xử lý ảnh nhập vào
input_path = "input"
for filename in os.listdir(input_path):
    if filename.endswith('.jpg'):
        file_path = os.path.join(input_path, filename)
        detect.face_detect(file_path, "portrait")
portrait_path = "process/portrait"
for subdir in os.listdir(portrait_path):
    subdir_path = os.path.join(portrait_path, subdir)
    compress.compress_faces(subdir_path, "portrait")

# Tiến hành so sánh
know_path = "embeddings/portrait/input.npz"
candidate_dir_path = "embeddings/face"
output_path = "output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
for filename in os.listdir(candidate_dir_path):
    if filename.endswith('.npz'):
        file_path = os.path.join(candidate_dir_path, filename)
        imgname = filename.replace('.npz', '.jpg')
        if compare.check(know_path, file_path):
            image_path_in_library = os.path.join(image_library_path, imgname) 
            output_image_path = os.path.join(output_path, imgname)
            shutil.copy2(image_path_in_library, output_image_path)
            print(f"Đã sao chép {imgname} vào thư mục output")
