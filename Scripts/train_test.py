from ultralytics import YOLO
import torch
import cv2
from os import listdir
import os
import numpy as np

TRAIN = False
INFER_FOLDER = True
PRINT_SINGLE = False

def main():

    if TRAIN:
        #model = YOLO('yolov8n.yaml', task='detect')
        model = YOLO('yolov8s.yaml', task='detect')

        model.train(
            data="data.yaml",
            epochs=200,
            batch=32,
            device=0,
            pretrained=False,
            name="model"
        ) 

    if INFER_FOLDER:
        MODEL_PATH = f'<synthetic_weights>'
        #MODEL_PATH = f'<real_weights>'

        PATH = f'<challenge_1>'
        #PATH = f'<challenge_2>'
        #PATH = f'<challenge_3>'
        #PATH = f'<challenge_4>'

        files = listdir(PATH)
        files = [f for f in files if os.path.isfile(f'{PATH}/{f}')]

        model = YOLO(MODEL_PATH)

        incorrect = []

        correct = 0
        for f in files:
            path = f'{PATH}/{f}'
            res = model(path)

            annotated = res[0].plot()
            resized = cv2.resize(annotated, (900, 1200))
            #resized = cv2.resize(annotated, (1920, 1080))
            cv2.imshow("Prediction", resized)
            cv2.waitKey(0)  # waits until you press a key
            cv2.destroyAllWindows()

            if res:
                if len(res) == 1:
                    if res[0].boxes:
                        correct += 1
                    else:
                        
                        """ annotated = res[0].plot()
                        resized = cv2.resize(annotated, (900, 1200))
                        #resized = cv2.resize(annotated, (1920, 1080))
                        cv2.imshow("Prediction", resized)
                        cv2.waitKey(0)  # waits until you press a key
                        cv2.destroyAllWindows() """

                        incorrect.append(path)
                else:
                    """ annotated = res[0].plot()
                    resized = cv2.resize(annotated, (900, 1200))
                    #resized = cv2.resize(annotated, (1920, 1080))
                    cv2.imshow("Prediction", resized)
                    cv2.waitKey(0)  # waits until you press a key
                    cv2.destroyAllWindows() """

                    incorrect.append(path)
            else:

                """ annotated = res[0].plot()
                resized = cv2.resize(annotated, (900, 1200))
                #resized = cv2.resize(annotated, (1920, 1080))
                cv2.imshow("Prediction", resized)
                cv2.waitKey(0)  # waits until you press a key
                cv2.destroyAllWindows() """

                incorrect.append(path)

        #print(incorrect)

        acc = correct / len(files)
        
        print(f'Acc.: {acc * 100}%')
        print(f'Total Correct: {correct}')
        print(f'Total Incorrect: {len(incorrect)}')
        print(f'Total Files: {len(files)}')

    if PRINT_SINGLE:
        MODEL_PATH = f'<model>'
        PATH = f'<photo_dir>'

        files = listdir(PATH)
        files = [f for f in files if os.path.isfile(f'{PATH}/{f}')]
        files = np.array(files)

        np.random.shuffle(files)

        model = YOLO(MODEL_PATH)

        for img in files:
            results = model(f'{PATH}/{img}')
            print(results)
            print(results[0])
            annotated = results[0].plot()
            resized = cv2.resize(annotated, (900, 1200))
            #resized = cv2.resize(annotated, (1920, 1080))
            cv2.imshow("Prediction", resized)
            cv2.waitKey(0)  # waits until you press a key
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()