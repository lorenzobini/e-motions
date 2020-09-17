import numpy as np
import cv2

from sklearn.model_selection import StratifiedShuffleSplit


def import_images(list_images, data_images, dim):
    """Imports images from path, adds to list,
    and converts to array of shape dim"""

    with open(list_images, 'r') as train_list:
        images = []
        labels = []

        for line in train_list:
            img = cv2.imread(data_images + line[0:len(line)-1])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # do we need to do this?
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            images.append(img)
            labels.append(line[0:len(line)-9]) #cutting "_123.jpg/n" from name string
    
    # convert to array
    images = np.array(images)

    # list of unique classes
    classes, labels = np.unique(np.array(labels), return_inverse=True)

    return images, labels, classes


def import_videos(list_videos, data_videos, dim, i0, i1, n):
    """Imports videos from path for videos on lines i0 to i1, returns array of middle frames,
     array of optical flow for n frames, array of labels, and unique classes"""

    with open(list_videos, 'r') as readme:
        lines = [line.rstrip('\n') for line in readme]

        m_frames = [] # middle frames only
        optical_flow = []
        labels = []
        classes = []

        # create data from set
        category_n = 0
        for line in lines[i0:i1]: # lines containing indices for set
            category, numbers = line.split("-> {")

            category = ''.join(category.split()) # remove whitespace
            classes.append(category)

            numbers = ''.join(numbers.split("}")) # remove curly bracket
            numbers = numbers.split(",")

            for num in numbers:
                num = num.zfill(4)
                vid = category + "_" + num + ".avi"
                data_path = data_videos + vid
                
                # initialize the video stream
                cap = cv2.VideoCapture(data_path)

                # calculate middle frame number
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                middle_frame = round(total_frames/2)

                # exit if there is no logical amount of frames
                if total_frames <= 0:
                    continue

                # initialize flow sequence
                flow_video = np.zeros((dim[0], dim[1], n))

                # grab first frame & use as starting point for optical flow
                (grabbed, frame1) = cap.read()
                prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

                # we're only using a subset of the frames; track index
                i = 0
                while True:
                    (grabbed, frame2) = cap.read()

                    # if the frame was not grabbed for some reason: exit
                    if not grabbed:
                        break

                    # for first n/2 frames, calculate optical flow
                    # n/2 because every frame will result in 2 channels (x,y) optical flow
                    if i < (n/2):
                        # convert frame to grayscale
                        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                        
                        # calculate optical flow
                        flow = cv2.calcOpticalFlowFarneback(
                            prev = prvs,
                            next = next,
                            flow = None, 
                            pyr_scale = 0.5,
                            levels = 1,
                            winsize = 15,
                            iterations = 3,
                            poly_n = 5,
                            poly_sigma = 1.2,
                            flags = 0
                            )
                        
                        # resize frame to preset dims
                        flow = cv2.resize(flow, dim, interpolation = cv2.INTER_AREA)
                        
                        # append to list
                        flow_video[:,:,i+i] = flow[:,:,0]
                        flow_video[:,:,i+i] = flow[:,:,1]

                        # set current frame as previous frame for next iteration
                        prvs = next
                    
                    # if frame == middle frame: add to m_frames
                    if i == middle_frame:
                        frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)     
                        
                        m_frames.append(frame)
                    
                    i += 1

                # append flow to total optical flow
                flow_video = np.array(flow_video)
                optical_flow.append(flow_video)

                # add category number to labels
                labels.append(category_n)
        
        category_n += 1
        
    # convert to numpy arrays
    m_frames = np.array(m_frames)
    optical_flow = np.array(optical_flow)
    labels = np.array(labels)
    classes = np.array(classes)

    return m_frames, optical_flow, labels, classes


def split_validation(x, y, p):
    """Splits dataset (x, y) using stratification:
    proportion p is validation set, 1-p is training set"""

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, validation_index in sss.split(x, y):
        train_x, val_x = x[train_index], x[validation_index]
        train_y, val_y = y[train_index], y[validation_index]

    return train_x, val_x, train_y, val_y

def split_validation_video(x1, y1, x2, y2, p):
    """Splits dataset (x, y) using stratification:
    proportion p is validation set, 1-p is training set"""

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, validation_index in sss.split(x1, y1):
        train_x1, val_x1 = x1[train_index], x1[validation_index]
        train_y1, val_y1 = y1[train_index], y1[validation_index]
        train_x2, val_x2 = x2[train_index], x2[validation_index]
        train_y2, val_y2 = y2[train_index], y2[validation_index]

    return train_x1, val_x1, train_y1, val_y1, train_x2, val_x2, train_y2, val_y2