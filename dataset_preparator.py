'''
Created on Jan 6, 2018

@author: Samer Saber
'''
import glob
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
def prepare_dataset(vehicle_paths, non_vehicle_paths, histogram = False):
    data_X = []
    data_Y = []
    c = 0
    for path in vehicle_paths :
        input_images = sorted(glob.glob(path+'/*.png'))
        for image in input_images:
            if c%2 :
                cvImg = cv2.imread(image)
                cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
                data_X.append(cvImg)
                data_Y.append(1)
            c += 1
    c = 0            
    for path in non_vehicle_paths :
        input_images = sorted(glob.glob(path+'/*.png'))
        for image in input_images:
            if c%2 :
                cvImg = cv2.imread(image)
                cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
                data_X.append(cvImg)
                data_Y.append(0)
            c += 1
                
    if (histogram) :    
        plt.figure()
        plt.hist(data_Y, bins=10)
        plt.title("Training dataset Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
    return  data_X, data_Y
        
if __name__ == '__main__':
    vehicle_paths = []
    non_vehicle_paths = []
    vehicle_paths.append("./dataset/vehicles/GTI_Far")
    non_vehicle_paths.append("./dataset/non-vehicles/GTI")
    data_x , data_y = prepare_dataset(vehicle_paths, non_vehicle_paths,True)
    dataset = (data_x,data_y)

#     dataset = shuffle(dataset)
#     
#     for data in dataset:
#         print (data[1])
#         cv2.imshow("img", data[0])
#         cv2.waitKey(0)
    
    pass