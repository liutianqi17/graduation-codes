import cv2


for num in range(20):
    vc = cv2.VideoCapture(r'E:\database\CBSR-Antispoofing\train_release/' + str(num+1) + r'\HR_1.avi') #读入视频文件
    rval=vc.isOpened()
    #timeF = 1  #视频帧计数间隔频率
    c = -1
    while rval:   #循环读取视频帧
        c = c + 1
        rval, frame = vc.read()
    #    if(c%timeF == 0): #每隔timeF帧进行存储操作
    #        cv2.imwrite('smallVideo/smallVideo'+str(c) + '.jpg', frame) #存储为图像
        if c == 0:
            continue
        if rval:
            #img为当前目录下新建的文件夹
            cv2.imwrite('img/' + str(num+1) + '_' + str(c) + '.jpg', frame) #存储为图像
            cv2.waitKey(1)
        else:
            break
    vc.release()