'''
Created on Jan 6, 2018

@author: Samer Saber
'''
import cv2
from feature_extraction import extract_features
from dataset_preparator import  prepare_dataset
import pickle
import numpy as np
from sklearn.utils import shuffle
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

def search_windows(img, windows):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]   
        #4) Extract features for that window using single_img_features()
        prediction = predict_window("./model.pkl",test_img)
        
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def predict_window(model_file, image):
    image = cv2.resize(image, (64, 64)) 
    feature = extract_features(image, color_space="HSV", hog_channel='ALL')
    
    with open(model_file, 'rb') as f:
        model, scalar = pickle.load(f)

    feature = np.array(feature).astype(np.float64)
    
    scaled_X = scalar.transform(feature.reshape(1, -1))

    
    prediction = model.predict(scaled_X) 
    
    return prediction

def process_frame(image, Debug_image = False):
    

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[int(image.shape[0]/2), int(image.shape[0])], 
                    xy_window=(100, 100), xy_overlap=(0.7, 0.7))
                       
    on_windows =  search_windows(image, windows)
    window_img = draw_boxes(image, on_windows, color=(0, 0, 255), thick=6)                    
    if (Debug_image) :
        plt.imshow(window_img)
        plt.show()
    return window_img
if __name__ == '__main__':
    

    
    static_image = False
    
    if static_image:
        #Calibrate the camera 
        frame = mpimg.imread('./test_images/test3.jpg')
        process_frame(frame, True)
        
    else:
        
    
        white_output = './project_video_out.mp4'
        clip1 = VideoFileClip("./project_video.mp4")
#         clip1 = VideoFileClip("./project_video.mp4").subclip(40,50)
        
        #clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
        white_clip = clip1.fl_image(process_frame) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
    
    pass