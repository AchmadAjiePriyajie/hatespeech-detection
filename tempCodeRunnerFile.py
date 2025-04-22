from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib

app = Flask(__name__)

saved_model = joblib.load('svm_model.joblib')
saved_tfidf = joblib.load('tfidf_vectorizer.joblib')

def predictNewData(tweets):

    vectorized_tweets = saved_tfidf.transform([tweets])
    input_prediction = saved_model.predict(vectorized_tweets)

    if input_prediction == 1:
        prediction = 'Ujaran Kebencian'
    elif input_prediction == 2:
        prediction = 'Kata kasar'
    elif input_prediction == 3:
        prediction = 'Kata kasar dan ujaran kebencian'
    else:
        prediction = 'Kata aman'

    return prediction

def labelCSVData(input_file):

    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_file)

    nama_kolom_baru = ['Teks']
    data = data.rename(columns=dict(zip(data.columns, nama_kolom_baru)))

    def map_prediction(value):
        if value == 1:
            return 'Ujaran Kebencian'
        elif value == 2:
            return 'Kata Kasar'
        elif value == 3:
            return 'Kata Kasar dan Ujaran Kebencian'
        else:
            return 'Kata Aman'

    # Perform prediction and label the data
    data['prediction'] = saved_model.predict(saved_tfidf.transform(data['Teks']))
    data['prediction'] = data['prediction'].apply(map_prediction)

    # Save the labeled data into a new CSV file
    labeled_file = 'dataset_indonesiam_toxic.csv'
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
        return send_file(labeled_file, as_attachment=True, download_name='dataset_indonesiam_toxic.csv')
    else:
        return render_template("content.html", error_message="Invalid file, please upload a CSV file.")
    

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
