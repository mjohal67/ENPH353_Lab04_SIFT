#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        #Numerical variables
        self._cam_id = 0 #/dev/video0
        self._cam_fps = 20
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)

        
        #camera info capture properties
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)
        #connect signal (button clicked) to slot (function to call upon execution)
        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320) #width
        self._camera_device.set(4, 240) #height

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        #SIFT and FLANN matching
        self.sift = cv2.SIFT_create()
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks=50)

        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.MIN_MATCH_COUNT = 10
        
        #self.chosen_image = None

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog() #open file dialogue
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0] #absolute path to selected file

        self.chosen_image = cv2.imread(self.template_path)
        frame_gray = cv2.cvtColor(self.chosen_image, cv2.COLOR_BGR2GRAY)
        self.kp1, self.des1 = self.sift.detectAndCompute(frame_gray, None) #no mask
        output_frame = self.chosen_image
        output_frame = cv2.drawKeypoints(frame_gray, self.kp1, output_frame)

        pixmap_features = self.convert_cv_to_pixmap(output_frame)
        self.template_label.setPixmap(pixmap_features)
        print("Loaded template image file: " + self.template_path)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
            self.live_image_label.clear() #clear image when camera feed off
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                     bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)


    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        

        #FLANN:https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        #FLANN and homography: https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
        #get second image keypoints
        kp2, des2 = self.sift.detectAndCompute(frame,None)
        matches = self.flann.knnMatch(self.des1, des2, k=2)

        good = []
        for m,n in matches: #two closest matches
            if m.distance < 0.7*n.distance: #if matches are far enough apart
                good.append(m)

        if len(good)>self.MIN_MATCH_COUNT:
            src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w,c = self.chosen_image.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            img2 = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT) )
            matchesMask = None


        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 2)

        matched_image = cv2.drawMatchesKnn(self.chosen_image,self.kp1,frame,kp2,matches,None,**draw_params)

        pixmap = self.convert_cv_to_pixmap(matched_image)
        self.live_image_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
