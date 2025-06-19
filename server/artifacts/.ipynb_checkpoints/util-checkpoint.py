# import json
# import pickle
# import numpy as np

# __locations = None
# __data_columns = None
# __model = None

# def get_estimated_price(location,sqft,bhk,bath):
#     try:
#         location = location.lower()
#         loc_index = __data_columns.index(location)
#     except:
#         loc_index = -1
        
#     x = np.zeros(len(__data_columns))
#     x[0] = sqft
#     x[1] = bath
#     x[2] = bhk
#     if loc_index >= 0:
#         x[loc_index] = 1
        
#     return round(__model.predict([x])[0],2)


# def get_location_names():
#     return __locations

# def load_saved_artifacts():
#     print("loading saved artifacts.... start")
#     global __data_columns
#     global __locations
#     global __model

#     with open("./artifacts/columns.json", "r") as f:
#         __data_columns = json.load(f)['data_columns']
#         __locations = __data_columns[3:]

#     with open("./artifacts/banglore_home_prices_model.pickle", "rb") as f:
#         __model = pickle.load(f)
#     print("loading saved artifacts.... done")
    
# if __name__ == '__main__':
#     load_saved_artifacts()
#     print(get_location_names())
#     print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
#     print(get_estimated_price('Indira Nagar', 1000, 2, 2))


import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        if __model is None or __data_columns is None:
            raise ValueError("Model or data columns not loaded. Call load_saved_artifacts() first.")
        
        location = location.lower()
        try:
            loc_index = __data_columns.index(location)
        except ValueError:
            loc_index = -1
            
        x = np.zeros(len(__data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1
            
        # Print input features for debugging
        print(f"Input features vector shape: {x.shape}")
        print(f"Features: {x}")
        
        predicted_price = __model.predict([x])
        
        # Check the predicted price
        print(f"Predicted price array: {predicted_price}")
        if predicted_price.size == 0:
            raise ValueError("Model returned an empty prediction.")
        
        print(f"Raw prediction: {predicted_price[0]}")
        
        return round(predicted_price[0], 2)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

def get_location_names():
    if __locations is None:
        load_saved_artifacts()
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __data_columns
    global __locations
    global __model

    try:
        with open("./artifacts/columns.json", "r") as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]
        print("Columns loaded successfully")
        
        with open("./artifacts/banglore_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)
        print("Model loaded successfully")
        
        # Test the model
        test_prediction = get_estimated_price('1st Phase JP Nagar', 1000, 3, 3)
        print(f"Test prediction: {test_prediction}")
        return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False
    
if __name__ == '__main__':
    success = load_saved_artifacts()
    if success:
        print("\nAvailable locations:")
        print(get_location_names())
        
        print("\nSample predictions:")
        print("1st Phase JP Nagar, 1000 sqft, 3 bhk, 3 bath:", 
              get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
        print("Indira Nagar, 1000 sqft, 2 bhk, 2 bath:", 
              get_estimated_price('Indira Nagar', 1000, 2, 2))
    else:
        print("Failed to load artifacts")



        










    