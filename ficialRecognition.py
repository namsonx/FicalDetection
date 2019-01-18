import os
import math
import face_recognition
import cv2
import pickle
import argparse
import msvcrt
from sklearn import neighbors
from face_recognition.face_recognition_cli import image_files_in_folder
from FaceRecognitionKnn import train
from time import sleep

train_dir = "C:\\Users\\mas2hc\\Desktop\\Smartcity\\data\\FaceDetection"
model_path = "C:\\Users\\mas2hc\\Desktop\\Smartcity\\data\\FaceDetection\\trained_knn_model.clf"
#key_press = 0

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)   
#             if len(face_bounding_boxes) == 0:
#                 print "There is no face in pic: ",  img_path
#             if len(face_bounding_boxes) >= 1:
#                 print 'There are more than 1 face in pic: ', img_path
            if len(face_bounding_boxes) != 1:
                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        print("Chose n_neighbors automatically:", n_neighbors)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

def create_model(train_dir, model_path):
    print("Training KNN classifier...")
    classifier = train(train_dir, model_save_path = model_path, n_neighbors=2)
    print("Training complete!")
    print type(classifier)
    return classifier

# def key_event():
#     global key_press
#     while True:
#         if msvcrt.kbhit():
#             print 'Key is pressed'
#             key_press = 1
#             return key_press

def save_image(name):
#     global key_press
#     t = threading.Thread(target=key_event)
#     t.start()
    img_folder = train_dir + '\\' + name
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    index = 1
    while True:
        sleep(0.2)
        print "Started capture pics. Press any key to start build train model"
        if msvcrt.kbhit():
            print 'Ended saving pics'
            break
        pic = cv2.VideoCapture("http://service:smartcity123@192.168.1.112/snap.jpg?JpegSize=L")
        ret, frame = pic.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0 or len(face_locations) > 1:
            continue
        img_path = img_folder + "\\" + name + "_" + str(index) + ".jpg"
        status = cv2.imwrite(img_path, frame)
        print "save image number %s status: %s." %(index, status)
        index = index + 1
        

def main():
    print 'starting main function'        
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', help='Running train mode, name of person who is going to train have to input')
    args = parser.parse_args()
    args.train='test'
    
    if args.train:
        save_image(args.train)
        #create_model(train_dir, model_path)
    
    if args.train==None:
        video_capture = cv2.VideoCapture(0)
        while True:
            print "Loop running"
            distance_threshold = 0.45
            ret, frame = video_capture.read()
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 0:
                continue
            faces_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)
                
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]
            for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_locations, are_matches):
                if rec:
                    print pred, loc
                else:
                    print "unkown", loc

if __name__ == "__main__":
    #create_model(train_dir, model_path)
    #save_image()
    main()
    pass
