from flask import Flask,render_template,request
import pandas as pd
import pickle
app = Flask(__name__)
data=pd.read_csv('Cleaned_dataset.csv')
model=pickle.load(open('price.pkl','rb'))

@app.route('/')
def home():
    locations = sorted(data['location'].unique())
    return render_template('loan.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
       
        locations = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath=float(request.form.get('bath'))
        total_sqft=float(request.form.get('total_sqft'))
        
        input_data=pd.DataFrame([[locations,bhk,bath,total_sqft]],columns=['location','bhk','bath','total_sqft'])
        
        prediction = model.predict(input_data)[0]
        prediction =(round(prediction,0))*1e5
        return render_template('loan.html',predict=prediction)
if __name__=='__main__':
    app.run(debug=True,port=5002)
