from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [float(N), float(P), float(K), float(temp), float(humidity), float(ph), float(rainfall)]
    print('DEBUG: Raw input feature_list =', feature_list)
    single_pred = np.array(feature_list).reshape(1, -1)
    print('DEBUG: single_pred (as array) =', single_pred)

    final_features = ms.transform(single_pred)
    print('DEBUG: After MinMaxScaler:', final_features)
    prediction = model.predict(final_features)
    print('DEBUG: Model prediction:', prediction)
    print('DEBUG: prediction[0] =', prediction[0], 'type:', type(prediction[0]))

    # If model predicts 0-based class, add 1 for crop_dict lookup
    predicted_class = int(prediction[0]) + 1
    print('DEBUG: predicted_class (for crop_dict) =', predicted_class)

    # Andhra Pradesh specific crops
    crop_dict = {
        1: "Rice", 
        2: "Maize", 
        3: "Jowar", 
        4: "Bajra", 
        5: "Ragi", 
        6: "Red Gram", 
        7: "Green Gram", 
        8: "Black Gram", 
        9: "Bengal Gram", 
        10: "Groundnut", 
        11: "Sesamum", 
        12: "Sunflower", 
        13: "Mango", 
        14: "Banana", 
        15: "Papaya", 
        16: "Guava", 
        17: "Coconut", 
        18: "Cashew", 
        19: "Citrus", 
        20: "Grapes", 
        21: "Cotton", 
        22: "Sugarcane"
    }

    if predicted_class in crop_dict:
        crop = crop_dict[predicted_class]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html', result=result)


# python main
if __name__ == "__main__":
   app.run(debug=True, port=5000, host='0.0.0.0')
