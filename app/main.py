import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


#load data
df = pd.read_csv("data/imdb.csv")



#split data
X_train , X_test , Y_train , Y_test = train_test_split(df["text"], df["label"] ,test_size = 0.2, random_state = 142)

#build pipeline
pipeline =Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model",MultinomialNB())
])

#train
pipeline.fit(X_train,Y_train)

#evaluate 
y_pred = pipeline.predict(X_test)
print("Acuracy : " , accuracy_score(Y_test ,y_pred))

joblib.dump(pipeline , "model/sentiment_model.pkl")