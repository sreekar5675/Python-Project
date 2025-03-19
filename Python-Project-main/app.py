from flask import Flask, render_template , request 
import pickle
import numpy as np 

app = Flask(__name__)
@app.route('/')
def Prediction():
    return render_template('Heart.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        htl=[float(x) for x in request.form.values()]
        model = pickle.load(open('health_prediction.pkl','rb'))    #pickle file load
        result=[np.array(htl)] # array convert
        prediction = model.predict(result)
        predicted_class = prediction[0]
        if predicted_class == 1:
            message = "Heart-Disease Patient: YES"
        else:
            message = "Heart-Disease Patient: NO"
    return render_template('Heart.html', prediction=message)
  

if __name__ == "__main__":
    app.run(debug=True)