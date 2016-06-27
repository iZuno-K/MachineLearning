import numpy as np
import cv2
import time

threshold = 15
sampling_times = 20

if __name__  == "__main__":
    cap = cv2.VideoCapture(0)
    switch1 = False
    switch2 = False
    match_data = []
    wname = 'frame'
    x_ratio = 3
    y_ratio = 5
    font = cv2.FONT_HERSHEY_PLAIN
    count = 0 #learning times

    sample_des = []
    learned_data = []
    class_names = []

    while(True):
        ret,src_img = cap.read()
        ysize = src_img.shape[0]
        xsize = src_img.shape[1]
        img = src_img[ysize/y_ratio:-ysize/y_ratio, xsize/x_ratio:-xsize/x_ratio]
        k = cv2.waitKey(5)

        orb = cv2.ORB()
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if k == 27: #Esc key
            break

        if k == ord('s'):
            switch1 = True
            switch2 = False
            start = time.time()

        if switch1:
            now = time.time()

            # carry out per 0.3 sec
            if (now - start) > 0.3:
                start = now
                # Initiate SIFT detector
                # orb = cv2.ORB()

                img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if count == 0:
                    sample_des = []
                # find the keypoints and descriptors with SIFT
                kp, des = orb.detectAndCompute(img1,None)
                sample_des.append(des)
                count += 1

                if count == sampling_times:
                    learned_data.append(sample_des)
                    count = 0
                    print "input class name"
                    class_names.append(raw_input())
                    switch1 = False
                    switch2 = True


            cv2.putText(img, str(count), (100, 100), font, 2, (0, 255, 0), 1, cv2.CV_AA)
            cv2.rectangle(src_img, (xsize/x_ratio, ysize/y_ratio), (xsize-xsize/x_ratio, ysize-ysize/y_ratio), (0,255,0), 2)

        if switch2:
            pre_min_distance = threshold
            class_index = -1

            src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

            kp,des = orb.detectAndCompute(src_gray, None)

            if isinstance(des, type(None)) != True:
                tmp_distance = []
                for i in range(len(learned_data)):
                    for j in learned_data[i]:
                        # Match descriptors.
                        if j.shape[1] == des.shape[1]:
                            matches = bf.match(j,des)
                            matches = sorted(matches,key = lambda x:x.distance)
                            if len(matches) != 0:
                                # print matches[0].distance
                                if matches[0].distance < pre_min_distance:
                                    pre_min_distance = matches[0].distance
                                    class_index = i


            if class_index != -1:
                cv2.putText(img, str(class_names[class_index]), (100, 100), font, 2, (0, 255, 0), 1, cv2.CV_AA)

        cv2.imshow(wname,src_img)

    cap.release()
    cv2.destroyAllWindows()
