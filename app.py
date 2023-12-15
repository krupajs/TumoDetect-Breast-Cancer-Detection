import numpy as np
import pandas as pd
from flask import Flask,render_template,request,redirect,url_for,jsonify
import pickle

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
app.secret_key="212"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/prediction',methods=['POST','GET'])

def prediction():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['radius_se',"texture_mean","smoothness_mean","compactness_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","smoothness_se","compactness_se","symmetry_se","fractal_dimension_se"]
    print(features_value)
    #df = pd.DataFrame(features_value)
    output = model.predict(features_value)
    #op='{0:.{1}f}'.format(prediction[0][1],2)
    print(output)
    if output[0]=='M':
        return render_template('predict.html', prediction_text='Cancer is Malignant')
    else:
        return render_template('predict.html', prediction_text='Cancer is Benign')
#input_features = [float(x) for x in request.form.values()]
#features_value = [[np.array(input_features)]]

  


    

'''
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction1=model.predict(features) 
    return render_template("predict.html",prediction_text="The patient is {}".format(prediction1))'''

if __name__=="__main__":
    app.run(debug=True)

'''
<video autoplay loop muted plays-inline class="back-video">
<source src="url_for{{('static',filename='Video/bgvid.mov')}} " type="video/mp4">
</video>'''
'''
@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        sno=request.form['serno']
        pwd=request.form['pwd']
        con=sqlite3.connect('login.db')
        con.row_factory=sqlite3.Row
        cur=con.cursor()
        cur.execute("select * from login where sno=? and password=?",(sno,pwd))
        data=cur.fetchone()
        print()
        if data:
            session['sno']=data['sno']
            return redirect('userpage')
        else:
            flash("service number and password mismatch",data)
    return redirect("/")


@app.route('/userpage',methods=['GET','POST'])
def userpage():
    return render_template('userpage.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('index')'''