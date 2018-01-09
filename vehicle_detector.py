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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


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
    feature = extract_features(image, color_space='YUV', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True)
    
    with open(model_file, 'rb') as f:
        model, scalar = pickle.load(f)

    feature = np.array(feature).astype(np.float64)
    
    scaled_X = scalar.transform(feature.reshape(1, -1))

    
    prediction = model.predict(scaled_X) 
    
    return prediction
count = 0
def process_frame(image, Debug_image = False):
    

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[int(390), int(550)], 
                    xy_window=(64, 64), xy_overlap=(0.85, 0.85))
    windows.extend( slide_window(image, x_start_stop=[None, None], y_start_stop=[int(390), int(image.shape[0])], 
                    xy_window=(150, 150), xy_overlap=(0.65, 0.65)))

    all_windows_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)     
    on_windows =  search_windows(image, windows)
    
    window_img = draw_boxes(image, on_windows, color=(0, 0, 255), thick=6)     
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,on_windows)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    

                   
    if (Debug_image) :
        fig = plt.figure()
        plt.subplot(221)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(222)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.subplot(223)
        plt.imshow(window_img)
        plt.subplot(224)
        plt.imshow(all_windows_img)
        plt.show()
    global count
    cv2.imwrite("./input/"+str(count)+".jpg", cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    cv2.imwrite("./output/"+str(count)+".jpg", cv2.cvtColor(draw_img,cv2.COLOR_BGR2RGB))
    count += 1
    return draw_img
if __name__ == '__main__':
    

    
    static_image = False
    count = 0
    if static_image:
        #Calibrate the camera 
        frame = mpimg.imread('./input/750.jpg')
        process_frame(frame, True)
        
    else:
        
    
        white_output = './project_video_out.mp4'
        clip1 = VideoFileClip("./project_video.mp4")
#         clip1 = VideoFileClip("./project_video.mp4").subclip(30,45)
        
        #clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
        white_clip = clip1.fl_image(process_frame) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
    
    pass