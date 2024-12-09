from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib

app = Flask(__name__)

saved_model = joblib.load('modelSVC.joblib')
saved_tfidf = joblib.load('TF-IDF.joblib')

def predictNewData(tweets):

    vectorized_tweets = saved_tfidf.transform([tweets])
    input_prediction = saved_model.predict(vectorized_tweets)

    if input_prediction == 1:
        prediction = 'Ujaran Kebencian'
    else:
        prediction = 'Bukan Ujaran Kebencian'

    return prediction

def labelCSVData(input_file):

    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_file)

    nama_kolom_baru = ['Teks']
    data = data.rename(columns=dict(zip(data.columns, nama_kolom_baru)))

    # Perform prediction and label the data
    data['prediction'] = saved_model.predict(saved_tfidf.transform(data['Teks']))
    data['prediction'] = data['prediction'].apply(lambda x: 'HS' if x == 1 else 'Non_HS')

    # Save the labeled data into a new CSV file
    labeled_file = 'ujaranKebencianApps.csv'
    data.to_csv(labeled_file, index=False)

    return labeled_file


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/content")
def content():
    return render_template("content.html")


@app.route("/predict", methods=["POST"])
def predict():
    tweets = request.form['tweets']
    if not tweets:
        return render_template("index.html", prediction_text="nullo")
    prediction = predictNewData(tweets)
    return render_template("content.html", prediction_text=prediction)

@app.route("/label", methods=["POST"])
def label():
    input_file = request.files['csv_file']
    if input_file and input_file.filename.endswith('.csv'):
        labeled_file = labelCSVData(input_file)
        return send_file(labeled_file, as_attachment=True, download_name='ujaranKebencianApps.csv')
    else:
        return render_template("content.html", error_message="Invalid file, please upload a CSV file.")
    

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
