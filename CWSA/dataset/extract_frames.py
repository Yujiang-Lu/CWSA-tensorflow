from face_utils import norm_crop, FaceDetector
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import sys

# splitnum = int(sys.argv[1])
# start = int(sys.argv[2])
# print(os.getcwd())
face_detector = FaceDetector()

face_detector.load_checkpoint("RetinaFace-Resnet50-fixed.pth")
torch.set_grad_enabled(False)


def extract_frames(video_path, mask_path, target_path, target_maskpath, scale):
    """
    Extract frames from a video. You can use either provided method here or implement your own method.

    params:
        - video_local_path (str): the path of video.
    return:
        - frames (list): a list containing frames extracted from the video.
    """
    ########################################################################################################
    # You can change the lines below to implement your own frame extracting method (and possibly other preprocessing),
    # or just use the provided codes.
    import cv2
    video_path_c23 = video_path.replace('raw', 'c23')
    video_path_c40 = video_path.replace('raw', 'c40')
    targetpath_c23 = target_path.replace('raw', 'c23')
    targetpath_c40 = target_path.replace('raw', 'c40')

    if not os.path.exists(targetpath_c23):
        os.makedirs(targetpath_c23)

    if not os.path.exists(targetpath_c40):
        os.makedirs(targetpath_c40)

    vid = cv2.VideoCapture(video_path)
    vid_c23 = cv2.VideoCapture(video_path_c23)
    vid_c40 = cv2.VideoCapture(video_path_c40)
    vid_mask = cv2.VideoCapture(mask_path)

    cap = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    # detect_per_video = 32
    i = 0
    count = 0
    s_x = 0
    s_y = 0
    max_l = 0
    num_frames = 0
    while True:
        success = vid.grab()
        success_c23 = vid_c23.grab()
        success_c40 = vid_c40.grab()
        success_mask = vid_mask.grab()
        if not success:
            break
        if i % 30 == 0:
        # if i % (cap // detect_per_video) == 0:
            success, frame = vid.retrieve()
            success_c23, frame_c23 = vid_c23.retrieve()
            success_c40, frame_c40 = vid_c40.retrieve()
            success_mask, frame_mask = vid_mask.retrieve()
            # cropping shown as below
            height, width, _ = frame.shape
            if frame is not None:
                boxes, landms = face_detector.detect(frame)
                if boxes.shape[0] == 0:
                    continue
                areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                order = areas.argmax()
                boxes = boxes[order]
                l, t, r, b = boxes.tolist()
                h = b - t
                w = r - l
                maxl = int(max(h, w) * scale)
                centerx = (t + b) / 2
                centery = (l + r) / 2
                startx = centerx - maxl // 2
                starty = centery - maxl // 2
                if startx <= 0:
                    startx = 0
                if startx + maxl >= height:
                    startx = height - maxl
                if starty <= 0:
                    starty = 0
                if starty + maxl >= width:
                    starty = width - maxl
                startx, starty = int(startx), int(starty)
                s_x = startx
                s_y = starty
                max_l = maxl
                # face = frame[startx:startx + maxl, starty:starty + maxl, :]
                # face_c23 = frame_c23[startx:startx + maxl, starty:starty + maxl, :]
                # face_c40 = frame_c40[startx:startx + maxl, starty:starty + maxl, :]
                # mask = frame_mask[startx:startx + maxl, starty:starty + maxl, :]
                # face = cv2.resize(face, (224, 224))
                # face_c23 = cv2.resize(face_c23, (224, 224))
                # face_c40 = cv2.resize(face_c40, (224, 224))
                # mask = cv2.resize(mask, (224, 224))
                # cv2.imwrite(target_path + '\\' + str(i) + '.png', face)
                # cv2.imwrite(targetpath_c23 + '\\' + str(i) + '.png', face_c23)
                # cv2.imwrite(targetpath_c40 + '\\' + str(i) + '.png', face_c40)
                # cv2.imwrite(target_maskpath + '\\' + str(i) + '.png', mask)
                # count += 1
                # if count >= detect_per_video:
                #     break
                # img = cv2.resize(img,(224,224))
        if i % 5 == 0:
            success, frame = vid.retrieve()
            success_c23, frame_c23 = vid_c23.retrieve()
            success_c40, frame_c40 = vid_c40.retrieve()
            success_mask, frame_mask = vid_mask.retrieve()
            if frame is not None:
                face = frame[s_x:s_x + max_l, s_y:s_y + max_l, :]
                face_c23 = frame_c23[s_x:s_x + max_l, s_y:s_y + max_l, :]
                face_c40 = frame_c40[s_x:s_x + max_l, s_y:s_y + max_l, :]
                mask = frame_mask[s_x:s_x + max_l, s_y:s_y + max_l, :]
                face = cv2.resize(face, (224, 224))
                face_c23 = cv2.resize(face_c23, (224, 224))
                face_c40 = cv2.resize(face_c40, (224, 224))
                mask = cv2.resize(mask, (224, 224))
                num_frames += 1
                cv2.imwrite(target_path + '\\' + str(num_frames) + '.png', face)
                cv2.imwrite(targetpath_c23 + '\\' + str(num_frames) + '.png', face_c23)
                cv2.imwrite(targetpath_c40 + '\\' + str(num_frames) + '.png', face_c40)
                cv2.imwrite(target_maskpath + '\\' + str(num_frames) + '.png', mask)
                count += 1
        i += 1
    vid.release()
    vid_c23.release()
    vid_c40.release()
    vid_mask.release()
    return frames
    ########################################################################################################


def extract_frames_real(video_path, target_path, scale):
    import cv2
    video_path_c23 = video_path.replace('raw', 'c23')
    video_path_c40 = video_path.replace('raw', 'c40')
    targetpath_c23 = target_path.replace('raw', 'c23')
    targetpath_c40 = target_path.replace('raw', 'c40')

    if not os.path.exists(targetpath_c23):
        os.makedirs(targetpath_c23)

    if not os.path.exists(targetpath_c40):
        os.makedirs(targetpath_c40)

    vid = cv2.VideoCapture(video_path)
    vid_c23 = cv2.VideoCapture(video_path_c23)
    vid_c40 = cv2.VideoCapture(video_path_c40)

    cap = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    # detect_per_video = 32
    i = 0
    count = 0
    s_x = 0
    s_y = 0
    max_l = 0
    num_frames = 0
    while True:
        success = vid.grab()
        success_c23 = vid_c23.grab()
        success_c40 = vid_c40.grab()
        if not success:
            break
        if i % 30 == 0:
        # if i % (cap // detect_per_video) == 0:
            success, frame = vid.retrieve()
            success_c23, frame_c23 = vid_c23.retrieve()
            success_c40, frame_c40 = vid_c40.retrieve()
            # cropping shown as below
            height, width, _ = frame.shape
            if frame is not None:
                boxes, landms = face_detector.detect(frame)
                if boxes.shape[0] == 0:
                    continue
                areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                order = areas.argmax()
                boxes = boxes[order]
                l, t, r, b = boxes.tolist()
                h = b - t
                w = r - l
                maxl = int(max(h, w) * scale)
                centerx = (t + b) / 2
                centery = (l + r) / 2
                startx = centerx - maxl // 2
                starty = centery - maxl // 2
                if startx <= 0:
                    startx = 0
                if startx + maxl >= height:
                    startx = height - maxl
                if starty <= 0:
                    starty = 0
                if starty + maxl >= width:
                    starty = width - maxl
                startx, starty = int(startx), int(starty)
                s_x = startx
                s_y = starty
                max_l = maxl
        if i % 5 == 0:
            success, frame = vid.retrieve()
            success_c23, frame_c23 = vid_c23.retrieve()
            success_c40, frame_c40 = vid_c40.retrieve()
            if frame is not None:
                face = frame[s_x:s_x + max_l, s_y:s_y + max_l, :]
                face_c23 = frame_c23[s_x:s_x + max_l, s_y:s_y + max_l, :]
                face_c40 = frame_c40[s_x:s_x + max_l, s_y:s_y + max_l, :]
                face = cv2.resize(face, (224, 224))
                face_c23 = cv2.resize(face_c23, (224, 224))
                face_c40 = cv2.resize(face_c40, (224, 224))
                num_frames += 1
                cv2.imwrite(target_path + '\\' + str(num_frames) + '.png', face)
                cv2.imwrite(targetpath_c23 + '\\' + str(num_frames) + '.png', face_c23)
                cv2.imwrite(targetpath_c40 + '\\' + str(num_frames) + '.png', face_c40)
                count += 1
        i += 1
    vid.release()
    vid_c23.release()
    vid_c40.release()
    return frames

# # fake video deal
# datapath = r'I:\compress_videos'
# targetpath = r'I:\FF++_c40_frame_1.5'
# scale = 1.5
#
# processdata = []
# i = 0
# for dataset in os.listdir(datapath):
#     if dataset == 'NeuralTextures':
#         datasetpath = datapath + '\\' + dataset
#         for c in ['c40']:
#             cpath = datasetpath + '\\' + c
#             for s in os.listdir(cpath):
#                 spath = cpath + '\\' + s
#                 for video in os.listdir(spath):
#                     if '.' in video:
#                         videopath = spath + '\\' + video
#                         maskpath = datasetpath + '\\mask\\masks\\videos\\' + video
#                         target_path = targetpath + '\\' + dataset + '\\' + c + '\\' + s + '\\' + video[:-4]
#                         target_maskpath = targetpath + '\\' + dataset + '\\mask\\' + video[:-4]
#                         processdata.append([videopath, maskpath, target_path, target_maskpath])
#                         i += 1
# print('total:', i)
# processdata.sort()
# processdatalen = len(processdata)
#

# splitnum = 1000
# start = 0
# while start != splitnum:
#     d = processdatalen // splitnum
#     if start != splitnum - 1:
#         data = processdata[start * d:(start + 1) * d]
#     else:
#         data = processdata[start * d:]
#     i = 0
#     datalen = len(data)
#
#     for video in data:
#         videopath = video[0]
#         maskpath = video[1]
#         target_path = video[2]
#         target_maskpath = video[3]
#         video_name = videopath.split('\\')[-1]
#         video_quality = videopath.split('\\')[-2]
#         video_split = videopath.split('\\')[-3]
#         video_dataset = videopath.split('\\')[-4]
#
#         if not os.path.exists(target_path):
#             os.makedirs(target_path)
#         # if len(os.listdir(target_path)) >= 500:
#         #     continue
#         if not os.path.exists(target_maskpath):
#             os.makedirs(target_maskpath)
#         extract_frames(videopath, maskpath, target_path, target_maskpath, scale)
#         i += 1
#         start += 1
#         # print(i, '\\', datalen, video_dataset, video_quality, video_split, video_name)
#         print(i, '\\', datalen, video_dataset, video_quality, video_split, video_name, start, '\\', splitnum)


# true video deal
datapath = r'I:\compress_videos'
targetpath = r'I:\FF++_c40_frame_1.5'
scale = 1.5

processdata = []
i = 0
for dataset in os.listdir(datapath):
    if dataset == 'Real':
        datasetpath = datapath + '\\' + dataset
        for c in ['c40']:
            cpath = datasetpath + '\\' + c
            for s in os.listdir(cpath):
                spath = cpath + '\\' + s
                for video in os.listdir(spath):
                    if '.' in video:
                        videopath = spath + '\\' + video
                        target_path = targetpath + '\\' + dataset + '\\' + c + '\\' + s + '\\' + video[:-4]
                        processdata.append([videopath, target_path])
                        i += 1
print('total:', i)
processdata.sort()
processdatalen = len(processdata)


splitnum = 1000
start = 0
while start != splitnum:
    d = processdatalen // splitnum
    if start != splitnum - 1:
        data = processdata[start * d:(start + 1) * d]
    else:
        data = processdata[start * d:]
    i = 0
    datalen = len(data)

    for video in data:
        videopath = video[0]
        target_path = video[1]
        video_name = videopath.split('\\')[-1]
        video_quality = videopath.split('\\')[-2]
        video_split = videopath.split('\\')[-3]
        video_dataset = videopath.split('\\')[-4]

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        extract_frames_real(videopath, target_path, scale)
        i += 1
        start += 1
        # print(i, '\\', datalen, video_dataset, video_quality, video_split, video_name)
        print(i, '\\', datalen, video_dataset, video_quality, video_split, video_name, start, '\\', splitnum)

