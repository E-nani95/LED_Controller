import dlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
detector = dlib.get_frontal_face_detector()
sp=dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec=dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
img_name="images1.jfif"
img_real=Image.open(img_name)
img_dlib=dlib.load_rgb_image(img_name)
faces=detector(img_dlib)

plt.imshow(img_dlib)
ax=plt.gca()
print("1. B-box \n2. Landmark")
choice=int(input(">> "))

if choice == 1:
    print("B-box크기: ex) 0.2")
    scale = float(input(">> "))

for face in faces:
    shape = sp(img_dlib, face)
    face_descriptor=facerec.compute_face_descriptor(img_dlib,shape)
    print(np.array(face_descriptor))

    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    match choice:
        case 1:
            new_w = int(w * (1 + scale))
            new_h = int(h * (1 + scale))
            #max로 경계를 벗어나지 않게 함
            new_x = max(x - int((new_w - w) / 2), 0)
            new_y = max(y - int((new_h - h) / 2), 0)

            # 박스 그리기
            rect = plt.Rectangle((new_x, new_y), new_w, new_h, fill=False, edgecolor='blue', linewidth=2)
            cropped = img_real.crop((new_x, new_y, new_x + new_w, new_y + new_h))

            ax.add_patch(rect)

            ax.add_patch(rect)
        case 2:

            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            for i in range(68):
                pt=shape.part(i)
                plt.plot(pt.x, pt.y, marker='o', markersize=2,color='yellow')

plt.axis("off")
plt.title("Detected Face with Landmarks")
plt.show()
match choice:
    case 1:
        cropped.save(f"{img_name}_cropped.jpg")