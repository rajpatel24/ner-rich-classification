import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def naive_byes_classification():
    """
    NER Rich Classifier using Naive Byes Algorithm
    """
    df = pd.read_csv("../Dataset/final.csv")

    cv = CountVectorizer()

    X = cv.fit_transform(df['sentence'] + ' ' + df['np'] + ' ' + df['vp'])
    y = df.ner_rich

    X = X.todense()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    ascore = accuracy_score(y_test, y_pred)
    print("\n\n>> Accuracy Score: ", ascore*100)

    pscore = precision_score(y_test, y_pred, average='binary')
    print("\n\n>> Precision Score: ", pscore*100)

    rscore = recall_score(y_test, y_pred, average='binary')
    print("\n\n>> Recall Score: ", rscore*100)

    # ------------------------------------------- Sample Data Prediction -----------------------------------------------

    df_test = pd.read_csv("../Dataset/test_sample.csv")

    input_data = cv.transform(df_test['sentence'] + ' ' + df_test['np'] + ' ' + df_test['vp'])

    predicted_output = nb.predict(input_data.toarray())

    original_output = df_test['ner_rich']

    print("\n\n ----------------------------------------------- \n")

    ascore = accuracy_score(original_output, predicted_output)
    print("\n\n>> Accuracy Score: ", ascore*100)

    pscore = precision_score(original_output, predicted_output, average='binary')
    print("\n\n>> Precision Score: ", pscore*100)

    rscore = recall_score(original_output, predicted_output, average='binary')
    print("\n\n>> Recall Score: ", rscore*100)


if __name__ == '__main__':
    naive_byes_classification()
