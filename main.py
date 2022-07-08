import argparse
import time
import cv2
import numpy as np
"""

视频播放过程中按s视频暂停，用鼠标框选追踪目标，空格继续

"""
#读入视频，实例化cv2.legacy.MultiTracker_create()
trackers = cv2.legacy.MultiTracker_create()
vs=cv2.VideoCapture("soccer_01.mp4")

while True:
    #一帧一帧知道结束，vs.read()[1]表示只取第二个返回值
    frame=vs.read()[1]
    if frame is None:
        break
    #每一帧太大了，利用cv2.resize（）等比缩放每一帧
    (h,w)=frame.shape[0:2]
    width=600
    r=width/float(w)
    dim=(width,int(h*r))
    frame=cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
    #追踪结果，并且会每一帧都更新.update()
    (success,boxes)=trackers.update(frame)
    #根据boxes的返回值，绘制图片
    for box in boxes:
        (x,y,w,h)=[int(v) for v in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #显示追踪到后的图片
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(100)&0xff#这步操作可以参考https://blog.csdn.net/hao5119266/article/details/104173400

    if key == ord("s"):
        #利用cv2.selectROI（）函数，选择一块区域
        box=cv2.selectROI("Frame",frame,fromCenter=False,showCrosshair=False)
        #利用cv2.legacy.TrackerKCF_create()实例化一个追踪器，并且把选择的box区域作为特征区域
        tracker=cv2.legacy.TrackerKCF_create()
        trackers.add(tracker,frame,box)

    #退出
    elif key==27:
        break

vs.release()
cv2.destroyAllWindows()


