
import cv2 #for video playing, grayscale to color conversions, rectangle drawing
import argparse #for video playing, directory finding
import os #for video playing, directory finding
import math
import numpy as np #for matrix operations
import time #for timing functions
import sys # for command line arguments

from distutils.core import setup
import py2exe

from distutils.filelist import findall
import matplotlib


#Comparing two frames
def compare_images(frame1dir, frame2dir):
    if not frame1dir or not frame2dir: print "compare_image :: Bad image detected"; return False
    frame1 =(cv2.imread(frame1dir))
    frame2 = (cv2.imread(frame2dir))

#Calculating useful image data
    if len(frame1) == len(frame2):

        total_pixels = len(frame1)
        pixel_rows = frame1.shape[0]
        pixel_columns = frame1.shape[1]
        image_increment = int(round(float(total_pixels) / 90))

    else:
        print "frame size mismatch"
        return False;

#Base image to augment

    frame_change = frame2

#Pixel checking loops
    for i in range(1,pixel_rows-1,3):
        for j in range(1,pixel_columns-1,3):
#Mean condition calculation by taking mean values of 3x3 pixels in each frame
            mean_condition= ((abs(frame2.item(i - 1, j,0) - frame1.item(i - 1, j,0)) + abs(
            frame2.item(i, j + 1,0) - frame1.item(i, j + 1,0)) +
            abs(frame2.item(i , j-1,0) - frame1.item(i, j-1,0)) + abs(frame2.item(i + 1, j,0) - frame1.item(i + 1, j,0)) + (abs(frame2.item(i , j,0) - frame1.item(i, j,0))
            )))/5

            #checking if mean condition at this location passes threshold, if it does mark a pixel box
            if (mean_condition > 5+image_increment):
                frame_change.itemset((i+1, j + 1,2), 255)

                frame_change.itemset((i, j - 1,2), 255)
                frame_change.itemset((i - 1, j,2), 255)
                frame_change.itemset((i, j + 1,2), 255)
                frame_change.itemset((i + 1, j,2), 255)
                frame_change.itemset((i, j,2), 255)
                frame_change.itemset((i-1,j-1,2),255)
                frame_change.itemset((i - 1, j + 1,2), 255)
                frame_change.itemset((i + 1, j - 1,2), 255)

    return frame_change #red marked frame

#checking sides and above values to create best size rectangle
def CheckDiagonals(frame_changes,x,y,total_rows,total_columns):
    bool = True #used to avoid repeats
    xmax = 0
    ymax = 0
   #initializing stack
    theStack = [(y, x)]
    while len(theStack) > 0 and (x<total_columns) and (y<total_rows): # loop checking stack coord neighbours
        y, x = theStack.pop() #getting coords from stack

        if (((frame_changes.item(y+1,x,2)) !=255 and (frame_changes.item(y,x+1,2)) != 255)): #if true, end of cluster

            return xmax,ymax #returning width and height

        if frame_changes.item(y+1, x, 2) == 255 and (bool):
            bool = False #not allowing repeat check
            ymax+=1 #increment height
            theStack.append((y+1,x)) #add next coord to stack to check

        else:
            bool = True
            xmax+=1 #increment width
            theStack.append((y,x+1)) #add next coord to stack to check



def cluster_find(boxes, overlapThresh):
   # if there are no boxes, return an empty list
   if len(boxes) == 0:
      return []

#convert ints to floats, important for divisions later on
   if boxes.dtype.kind == "i":
      boxes = boxes.astype("float")
#
   # initialize the list of picked indexes
   pick = []

   #coordinates of the bounding boxes
   x1 = boxes[:,0]
   y1 = boxes[:,1]
   x2 = boxes[:,2]
   y2 = boxes[:,3]

   # compute the area of the bounding boxes and sort the bounding
   # boxes by the bottom-right y-coordinate of the bounding box
   area = (x2 - x1 + 1) * (y2 - y1 + 1)

   idxs = np.argsort(y2) # quick sort by index


   # loop while some indexes still remain
   #this loop works by comparing rectangles, represented by indexs sorted by highest y2
   while len(idxs) > 0:
      # grab the last index in the indexes list and add the
      # index value to the list of picked indexes
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)

      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box, looping through indexes
      xx1 = np.maximum(x1[i], x1[idxs[:last]])
      yy1 = np.maximum(y1[i], y1[idxs[:last]])
      xx2 = np.minimum(x2[i], x2[idxs[:last]])
      yy2 = np.minimum(y2[i], y2[idxs[:last]])

      # width and height of the bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)

      # Ratio of overlap
      overlap = (w * h) / area[idxs[:last]]
      """
      print "Where: "
      print np.where(overlap > overlapThresh)[0]
      print " Concat: "
      print np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
      """

      # delete all indexes from the index list that do not fall within overlap threshold
      idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
      # return only the bounding boxes that were picked
   return boxes[pick].astype("int")

def imageLoop(imagestream):
#image file interaction setting paths to variables etc.
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'png'.")
    ap.add_argument("-o", "--output", required=False, default='ex6.avi', help="output video file")
    args = vars(ap.parse_args())


    file = os.path.dirname(__file__)
    print file
    selection = 'moving-localization/' + str(imagestream)

    dir_path = os.path.join(file,selection)
    print dir_path
    dir_path2 = os.path.normpath(dir_path)
    ext = args['extension']
    output = args['output']



#how it finds next file
    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)

    image_path = os.path.join(dir_path, images[0])
    # Determine the width and height from the first image
    dir_path2 = os.path.join(file, selection, images[0])

    regular_frame = cv2.imread(image_path)

    height, width, channels = regular_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

    #image opening loop

    for image in range(0,len(images)-1,1):

        rectangle_list = []
        #2 frames path bring compared in 1 iteration
        image_path = os.path.join(dir_path, images[image])
        image_path2 = os.path.join(dir_path, images[image+1])

        regular_frame = cv2.imread(image_path)
        #putting frame into matrix
        #calling comparison method
        frame_changes = (compare_images(image_path,image_path2))
        #calculations for getting image_increment
        total_pixels = len(frame_changes)
        pixel_rows = frame_changes.shape[0]
        pixel_columns = frame_changes.shape[1]
        image_increment = int(round(float(total_pixels) / 90)) #used for generalizing code, changes thresholds

        #iterating over each pixel
        for i in range(3,pixel_columns-7,5):
            for j in range(3,pixel_rows-7,5):
                #if red is found
                if frame_changes.item(j,i,2) ==255:
                    #calling cluster finding method
                    max = CheckDiagonals(frame_changes,i,j, pixel_rows-10, pixel_columns-10)

                    rec =0
                    if max == None:
                        max = [0,0]
                    xmax = max[1]
                    ymax = max[0]
                    if xmax + ymax > image_increment**2:
                        rectangle = i, j, i + ymax, j + xmax
                        #making rectangle list for big cluster finder
                        rectangle_list.append(rectangle)
        #rectangle list formatting
        rectangle_list = np.array(rectangle_list)
        rectangle_list.astype('float')
        #making elements floats for proper overlap calculations
        #calling cluster_find to delete overlapping/interior rectangles
        rectList = cluster_find(rectangle_list, 0)

        for p in rectList: #rectlist contains 4 coordinates for each rectangle
            #rectangle drawing function
            cv2.rectangle(regular_frame, (p[0], p[1]), (p[2], p[3]), (255, 255, 0), 2)

        #out.write(regular_frame)  # Write out frame to video

        cv2.imshow('video', regular_frame)
        #video player
        if (cv2.waitKey(1) & 0xFF) == ord('q'): #q to quit
            break

    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))


#timing calculations





