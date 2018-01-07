'''
Created on Jan 7, 2018

@author: Samer Saber
'''
import numpy as np
import cv2
from skimage.feature import hog

orient = 32
pix_per_cell = 16
cell_per_block = 2
hist_bins = 32
spatial_size = (16,16)


def bin_spatial(img, size=(16, 16)):
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def get_hog_features(img, orient=orient,
                     pix_per_cell=pix_per_cell,
                     cell_per_block=cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def convert_color(image, color_space='RGB'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)
    return feature_image

def extract_features(image, color_space='RGB', spatial_size=spatial_size,
                     hog_channel='ALL'):

    file_features = []
    feature_image = convert_color(image, color_space)

    spatial_features = bin_spatial(feature_image, size=spatial_size)
    file_features.append(spatial_features)

    # Apply color_hist()
    #hist_features = color_hist(feature_image, nbins=hist_bins)
    #file_features.append(hist_features)

    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                        pix_per_cell, cell_per_block, vis=False,
                                        feature_vec=True)
        # Append the new feature vector to the features list
    file_features.append(hog_features)
    assert (file_features != [])
    return np.concatenate(file_features)

if __name__ == '__main__':
    image = cv2.imread('./dataset/vehicles/GTI_Far/image0000.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    features = extract_features(image)
    f , im = get_hog_features(image[:,:,0], vis=True)   
#     print (features.shape)
    cv2.imshow('hog', im)
    cv2.waitKey(0)
    pass