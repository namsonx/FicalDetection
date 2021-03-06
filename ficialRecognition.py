import os
import math
import face_recognition
import cv2
import pickle
import argparse
import time
from sklearn import neighbors
from face_recognition.face_recognition_cli import image_files_in_folder
from time import sleep
import paho.mqtt.client as paho
from datetime import datetime

train_dir = os.getenv('TRAIN_DIR', '/home/bosch/Son/TrainDir')
model_path = train_dir + '/trained_knn_model.clf'
broker = os.getenv('BROKER', '192.168.1.31')
port = 1883
camera_ip = os.getenv('CAMERA_IP', '192.168.1.50')
cam_link = 'http://service:smartcity123@' + camera_ip + '/snap.jpg?JpegSize=L'
detection_delay_ms = 15 * 60 * 1000
current_milli_time = lambda: int(round(time.time() * 1000))
vistor = []

def on_publish(client, userdata, result):
    print("Published data")
    pass

def on_disconnect(client, userdata, rc):
    print("client disconnected ok")
    
client = paho.Client("fical-detection")
client.on_publish = on_publish

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)   
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

def save_image(name):
    img_folder = train_dir + '/' + name
    if not os.path.exists(img_folder):
	    os.mkdir(img_folder)
    index = 1
    try:
        while True:
            sleep(0.2)
            print "Started capture pics. Press key Ctrl+c to start build train model"
            pic = cv2.VideoCapture(cam_link)
            ret, frame = pic.read()
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 0 or len(face_locations) > 1:
                continue
            # Only 1 face in face_locations
            for face in face_locations:
                top, right, bottom, left = face
                face_img = frame[top:bottom, left:right]
            img_path = img_folder + "/" + name + "_" + str(index) + ".jpg"
            status = cv2.imwrite(img_path, face_img)
            print "save image number %s status: %s." %(index, status)
            index = index + 1
    except KeyboardInterrupt:
        print 'Ended collecting pics'
        pass
    
def update_image(name, face_image):    
    print 'Starting update image for: ', name
    current_time = str(datetime.now())
    current_time = current_time.replace(" ", "").replace(":", "").replace("-", "")
    current_time = current_time.split(".")[0]
    img_folder = train_dir + '/' + name
    img_path = img_folder + '/' + 'update_face_' + current_time + '.jpg'
    status = cv2.imwrite(img_path, face_image)
    print "save image %s status: %s." %(img_path, status)
    
def main():
    print 'starting main function'        
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', help='Running train mode, name of person who is going to train have to input')
    args = parser.parse_args()
    
    if args.train:
        save_image(args.train)
        create_model(train_dir, model_path)
    
    if args.train==None:
        try:
            while True:
                print "Loop running"
                person = {}
                video_capture = cv2.VideoCapture(cam_link)
                distance_threshold = 0.3
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
                        name = str(pred)
                        print name, loc
                        top, right, bottom, left = loc
                        face = frame[top:bottom, left:right]
                        update_image(name, face)
                        check = 0
                        for p in vistor:
                            if p['name']==name:
                                check = 1
                                now = current_milli_time()
                                if now - p['record_time'] >= detection_delay_ms:
                                    display_name = p['name']
                                    p['record_time'] = current_milli_time()
                                    client.connect(broker, port)
                                    client.publish("test/detection", display_name)
                        if check==0:
                            person['name'] = name
                            person['record_time'] = current_milli_time()
                            vistor.append(person)
                            display_name = person['name']
                            client.connect(broker, port)
                            client.publish("test/detection", display_name)
                    else:
                        print "unkown", loc
        except KeyboardInterrupt:
            print 'Stoped face detection application'
            pass

if __name__ == "__main__":
    main()
    pass
