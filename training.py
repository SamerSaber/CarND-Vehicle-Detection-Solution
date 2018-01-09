'''
Created on Jan 7, 2018

@author: Samer Saber
'''
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
import dataset_preparator
import numpy as np 
from feature_extraction import extract_features

def get_SVCmodel():
    model = LinearSVC()
    return model
    
def train(vehicle_paths, non_vehicle_paths):
    
    
    data_X , data_Y = dataset_preparator.prepare_dataset(vehicle_paths, non_vehicle_paths,False)
    
    print("Preprocess {} images".format(len(data_X)))
    features = []
    for image in data_X:
        features.append(extract_features(image, color_space='YUV', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True))
    
    features = np.array(features).astype(np.float64)
    
    scalar = StandardScaler().fit(features)
    scaled_X = scalar.transform(features)

    
    stat = np.random.randint(0, 100)
    scaled_X, y = shuffle(scaled_X, data_Y)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                        test_size=.3,
                                                        random_state=stat)
    
    model = get_SVCmodel() 
    print("Training...")
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    print("Testing score: {}".format(score))
    
    with open('model.pkl', 'wb') as f:
        pickle.dump((model,scalar), f)

    
        
if __name__ == '__main__':
    
    vehicle_paths = []
    non_vehicle_paths = []
    
    vehicle_paths.append("./dataset/vehicles/GTI_Far")
    vehicle_paths.append("./dataset/vehicles/GTI_Left")
    vehicle_paths.append("./dataset/vehicles/GTI_MiddleClose")
    vehicle_paths.append("./dataset/vehicles/GTI_Far")
    vehicle_paths.append("./dataset/vehicles/KITTI_extracted")
        
    non_vehicle_paths.append("./dataset/non-vehicles/Extras")
    non_vehicle_paths.append("./dataset/non-vehicles/GTI")
    
    train(vehicle_paths, non_vehicle_paths)
    pass