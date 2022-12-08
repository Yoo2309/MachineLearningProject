import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


def CheckComment(data):
    df1= pd.read_csv("pages/CheckSpamToxicComment/YoutubeSpamMergedData.csv")
    df1_data = df1[["CONTENT","CLASS"]]
    # Features and Labels
    df1_x = df1_data['CONTENT']
    df1_y = df1_data.CLASS

    # Extract Feature With CountVectorizer
    corpus1 = df1_x
    cv1 = CountVectorizer()
    X1 = cv1.fit_transform(corpus1) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, df1_y, test_size=0.25, random_state=42)

    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf1 = MultinomialNB()
    clf1.fit(X_train1,y_train1)
    clf1.score(X_test1,y_test1)


    #TOXIC COMMENT

    df2= pd.read_csv("pages/CheckSpamToxicComment/data_train_clean.csv")
    df2_data = df2[["clean_comment","toxic"]]
    # Features and Labels
    df2_x = df2_data['clean_comment']
    df2_y = df2_data.toxic

    # Extract Feature With CountVectorizer
    corpus2 = df2_x
    cv2 = CountVectorizer()
    X2 = cv2.fit_transform(corpus2) # Fit the Data

    from sklearn.model_selection import train_test_split
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, df2_y, test_size=0.25, random_state=42)

    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf2 = MultinomialNB()
    clf2.fit(X_train2,y_train2)
    clf2.score(X_test2,y_test2)

    data = [data]
    vect1 = cv1.transform(data).toarray()
    my_prediction1 = clf1.predict(vect1)

    vect2 = cv2.transform(data).toarray()
    my_prediction2 = clf2.predict(vect2)
    
    if (my_prediction1 == 0 and my_prediction2 == 0):
        return "Non Spam and Non Toxic"
    elif (my_prediction1 == 1 and my_prediction2 == 0):
        return "Spam and Non Toxic"
    elif (my_prediction1 == 0 and my_prediction2 == 1):
        return "Non Spam and Toxic"
    else:
        return "Spam and Toxic"