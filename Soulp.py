import numpy as np
import pandas as pd
import treetaggerwrapper as tt
from nltk.corpus import stopwords
from nltk.stem.porter import * 
from gensim.models.word2vec import Word2Vec
from sklearn.tree import DecisionTreeClassifier
from PyQt5.QtWidgets import *
from PyQt5.QtGui import * 
from PyQt5.QtCore import Qt
import pickle
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys


##################################################################################


#korisnicko sucelje
class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setStyleSheet("background-color: #FACBFF;")
        self.setWindowTitle("Soulp.py")
        self.setFixedWidth(700)
        self.setFixedHeight(600)

        #naslov 
        self.title = QLabel("SOULP.PY", self)
        self.title.move(0,0)
        self.title.resize(700, 200)
        self.title.setFont(QFont("Consolas", 30))
        self.title.setAlignment(Qt.AlignCenter)

        #prostor za upisivanje recenice
        sentence = QTextEdit(self)
        sentence.move(100, 180)
        sentence.resize(500, 120)
        sentence.setFont(QFont("Consolas", 15))
        sentence.setAlignment(Qt.AlignCenter)
        sentence.setStyleSheet(
            """
            background:rgb(255, 255, 255);
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(30)
        )

        #gumb
        self.button = QPushButton("Get sentiment", self)
        self.button.move(230, 310)
        self.button.resize(240, 80)
        self.button.setFont(QFont("Consolas", 15))
        self.button.setStyleSheet(
            """
            background-color: #F177FF;
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(30)
        )

        #prostor za rezultat
        sentiment = QLabel("", self)
        sentiment.move(200, 430)
        sentiment.resize(300, 80)
        sentiment.setFont(QFont("Consolas", 15))
        sentiment.setAlignment(Qt.AlignCenter)
        sentiment.setStyleSheet(
            """
            background:rgb(255, 255, 255);
            border-top-left-radius:{0}px;
            border-bottom-left-radius:{0}px;
            border-top-right-radius:{0}px;
            border-bottom-right-radius:{0}px;
            """.format(30)
        )

        #funkcija koja se izvrsava pritiskom na gumb
        def on_click():
            
            #ucitaj model i stablo
            model = Word2Vec.load("MyModel")
            treeFile = open("MyTree.pkl", 'rb')
            tree = pickle.load(treeFile)
            treeFile.close()
            senti = tree.predict(getTestVecs(model, [sentence.toPlainText()]))[0]

            if(senti == "positive"):
                sentiment.setStyleSheet(
                """
                background-color: #77FF9C;
                border-top-left-radius:{0}px;
                border-bottom-left-radius:{0}px;
                border-top-right-radius:{0}px;
                border-bottom-right-radius:{0}px;
                """.format(30))
            elif(senti == "neutral"):
                sentiment.setStyleSheet(
                """
                background-color: #CF77FF;
                border-top-left-radius:{0}px;
                border-bottom-left-radius:{0}px;
                border-top-right-radius:{0}px;
                border-bottom-right-radius:{0}px;
                """.format(30))
            elif(senti == "negative"):
                sentiment.setStyleSheet(
                """
                background-color: #7782FF;
                border-top-left-radius:{0}px;
                border-bottom-left-radius:{0}px;
                border-top-right-radius:{0}px;
                border-bottom-right-radius:{0}px;
                """.format(30))

            sentiment.setText(senti)

        self.button.clicked.connect(on_click)

        self.show()

"""app = QApplication(sys.argv)
window = Window()
sys.exit(app.exec())"""


##################################################################################


#dohvati tekst tweeta iz id-ja
def idToTweet(id):

    #informacije za pristup twitter api-ju
    consumer_key = "ceMbFjyJNgkcamGBvejG7a38S"
    consumer_secret = "KtD7hx6jsCrsyy8asG2S9YMSZClSqn4vvNOu93eYL6wEKbtbPr"
    token = "1602424396668407808-ldhsnlerOUloDT8rYQ2aOCEiGNbU7w"
    token_secret = "lNkTLABUWOSdjbgSTbgkAkj2ZtMzspCKSbpVeIShKcn3o"
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAMiCkQEAAAAA5%2FJxs%2FAnpJTPqjDsOk%2B9nWgGihE%3D0XvXLSdqmxakkGQyvCy9abGrraZIP9VnvO38ymtPmbbXpZWNo8"

    client = tweepy.Client(bearer_token=bearer_token, consumer_key=consumer_key, 
        consumer_secret=consumer_secret, access_token=token, 
        access_token_secret=token_secret)

    tweetObject = client.get_tweet(id)
    tweetText = str(tweetObject[0])

    return tweetText


#odredi sentiment pomocu Vadera
def vaderAnalize(tweet):

    vaderAnalyzator = SentimentIntensityAnalyzer() #analizator sentimenata
    sentimentDict = vaderAnalyzator.polarity_scores(tweet)
    
    #vrati ispravnu oznaku
    if(sentimentDict["neg"] > 0.1 or sentimentDict["pos"] > 0.1):
        if(sentimentDict["neg"] > sentimentDict["pos"]):
            return "negative"
        elif(sentimentDict["neg"] < sentimentDict["pos"]):
            return "positive"
        else:
            return "neutral"
    else:
        return "neutral"


#kreiraj skupove ciste skupove podataka
def prepareDataset(startpath, endpath, minline, maxline):

    lines = [] #lista za linije dokumenta
    ef = open(endpath, "a", encoding='utf-8') #dokument za skup podataka

    #1. Procitaj odreden broj tweetova 
    with open(startpath) as sf:
        for i, line in enumerate(sf):
            if i > maxline-1:
                break
            else:
                if i > minline:
                    lines.append(sf.readline())

    for l in lines:

        #2. Dohvati tekst tweeta preko id-ja
        tweetText = idToTweet(l[0:19])
        proTweet = " ".join(preprocess(tweetText))

        if(proTweet != "none" and len(proTweet) != 0):
            #3. Analiziraj tekst preko Vadera
            sentiment = vaderAnalize(tweetText)

            #4. Zapisi u dokument
            ef.write(sentiment + "," + proTweet + "\n")
 

##################################################################################


#procitaj podatke
def readDataset(path):

    f = open(path, "r", encoding='utf-8')
    lines = f.readlines()

    sentiments = [] #lista sentimenata
    tweets = [] #lista tweetova

    for l in range(0, len(lines)):
        lparts = lines[l].split(",")
        if(len(lparts) == 2 and len(lparts[1]) != 0):
            sentiments.append(lparts[0])
            tweets.append(lparts[1])

    return [sentiments, tweets]


##################################################################################


#procitaj podatke o vektorima
def readVecs(vecfilename):
    f = open(vecfilename, "r")
    lines = f.readlines()
    arrayVecs = []
    for l in lines:
        arrayVecs.append(l.split(","))
    floatVecs = []
    for arr in arrayVecs:
        floatArr = []
        for a in arr:
            floatArr.append(float(a))
        floatVecs.append(floatArr)
    return floatVecs


##################################################################################


#ocisti tekst tweeta
def preprocess(rawTweet):

    #pretvori tweet u listu elemenata
    listLines = rawTweet.split("\n")
    rawTweet = " ".join(listLines)
    listTweet = rawTweet.split(" ")
    listGood = [] #lista za dobre rijeci
    stops = set(stopwords.words("english")) #lista stop-rijeci

    #1. Makni korisnicka imena, linkove i stop-rijeci
    for a in listTweet:
        a = a.lower()
        if("@" not in a and "https" not in a and a != "rt" and a not in stops): 
            listGood.append(a)
    listTweet = listGood
    listGood = []

    #2. Makni interpunkcije, brojeve i posebne znakove
    for b1 in listTweet:
        
        elementGood = [] #lista za dobre znakove u pojedinoj rijeci
        
        for b2 in list(b1):
            if((ord(b2) >= 97 and ord(b2) <= 122) or b2 == "#"):
                elementGood.append(b2)

        if(len(elementGood) != 0):
            listGood.append("".join(elementGood))

    listTweet = listGood
    listGood = []

    #3. Normaliziraj rijeci
    
    #Lemmatization
    tagger = tt.TreeTagger(TAGLANG ='en', TAGDIR ='TreeTagger')
    for c in listTweet:
        tag = tagger.tag_text(c)
        listGood.append(tag[0].split('\t')[-1])

    listTweet = listGood
    listGood = []

    #Stemming
    stemmer = stemmer = PorterStemmer() 
    for c in listTweet:
        listGood.append(stemmer.stem(c))

    return listGood


##################################################################################


#vrati prosjecne vektore recenica
def getAvgVec(model, proTweets, filename = ""):

    vecs = [] #lista svih prosjecnih vektora

    for proT in proTweets:
            
        addVec = 0 #zbroj vektora rijeci u recenici
        nonExist = 0 #broj rijeci koje ne postoje u rijecniku
            
        for word in proT:
            if word in model.wv:
                addVec += model.wv[word]
            else:
                nonExist += 1
        
        if(len(proT) - nonExist != 0):
            avgVec = addVec/(len(proT) - nonExist) #prosjecni vektor recenice
        else:
            avgVec = [0]*100

        vecs.append(avgVec)

        if(filename != ""):
            f = open(filename, "a") #dokument za spremanje vektora
            for a in range(len(avgVec)): #zapisi u file
                f.write(str(avgVec[a]))
                if(a != len(avgVec)-1):
                    f.write(",")
            f.write("\n")

    return vecs


#izracunaj vektore za testne tekstove
def getTestVecs(model, rawTweets):
    if(isinstance(rawTweets, str)):
        vecs = getAvgVec(model, [rawTweets])
        return vecs[0]
    elif(isinstance(rawTweets, list)):
        vecs = getAvgVec(model, rawTweets)
        return vecs


##################################################################################


#analiziraj sentimente i zapisi ih u dokument
def getResults(startpath, endpath, maxline):

    lines = readDataset(startpath) #linije dokumenta
    #1. Odrezi interval linija
    sentiments = lines[0][0:maxline]
    tweets = lines[1][0:maxline]
    ef = open(endpath, "a", encoding='utf-8') #dokument za rezultate

    #2. Razdvoji tweetove na rijeci
    splitTweets = []
    for t in tweets:
        t = t[0:len(t)-1].split(" ")
        splitTweets.append(t)

    #3. Ucitaj model i stablo odluke
    model = Word2Vec.load("MyModel")
    treeFile = open("MyTree.pkl", 'rb')
    tree = pickle.load(treeFile)
    treeFile.close()

    #4. Izracunaj vektore za testne tekstove
    test = getTestVecs(model, splitTweets)

    #5. Predvidi rezultate
    predictions = tree.predict(test)

    #6. Pronadi postotke za sentimente i preciznost
    
    #brojaci za: positive-word2, neutral-word2, negative-word2, 
    #positive-vader, neutral-vader, negatve-vader, accuracy
    results = [0, 0, 0, 0, 0, 0, 0] 

    for p in range(len(predictions)):

        if(predictions[p] == 'positive'):
            results[0] += 1
        if(predictions[p] == 'neutral'):
            results[1] += 1
        if(predictions[p] == 'negative'):
            results[2] += 1
            
        if(sentiments[p] == 'positive'):
            results[3] += 1
        if(sentiments[p] == 'neutral'):
            results[4] += 1
        if(sentiments[p] == 'negative'):
            results[5] += 1

        if(predictions[p] == sentiments[p]):
            results[6] += 1

    #7. Zapisi u dokument
    ef.write("Predictions\n")
    ef.write("positive," + str(results[0]/len(predictions)) + "\n")
    ef.write("neutral," + str(results[1]/len(predictions)) + "\n")
    ef.write("negative," + str(results[2]/len(predictions)) + "\n")
    ef.write("\nVader\n")
    ef.write("positive," + str(results[3]/len(predictions)) + "\n")
    ef.write("neutral," + str(results[4]/len(predictions)) + "\n")
    ef.write("negative," + str(results[5]/len(predictions)) + "\n")
    ef.write("\naccuracy," + str(results[6]/len(predictions)) + "\n")

    print(str(results[6]/len(predictions)))


##################################################################################


#GLAVNA FUNKCIJA

def main():

    #1. Kreiraj ciste skupove podataka
    """prepareDataset("startpath", "endpath", 0, 3000)"""

    #2. Procitaj podatke
    """parts = readDataset("C://Users//Sugarplum//Desktop//luciFER//ZavrsniRad//Datasets//0sve//Clean0//ModelTrain.txt")
    sentiments = parts[0]
    tweets = parts[1]"""

    #3. Treniraj Word2Vec s podatcima i spremi model
    """splitTweets = []
    for t in tweets:
        t = t[0:len(t)-1].split(" ")
        splitTweets.append(t)

    model = Word2Vec(splitTweets, vector_size = 100, min_count=1, window=5, workers=3, sg=0)
    model.save("MyModel")"""

    #4. Izracunaj prosjecne vektore recenica i zapisi u dokument
    """vecs = getAvgVec(model, splitTweets, filename = "MyVecs.txt")"""

    #5. Treniraj stablo odluke sa sentimentima i spremi ga
    """tree = DecisionTreeClassifier(max_depth = 10)
    tree.fit(vecs, sentiments)
    tree_file = open("MyTree.pkl", "wb")
    pickle.dump(tree, tree_file)
    tree_file.close()"""

    #* Ucitaj model i procitaj vektore
    """model = Word2Vec.load("MyModel")
    vecs = readVecs("MyVecs.txt")"""
 
    #6. Testiraj model i premi podatke
    """getResults("C://Users//Sugarplum//Desktop//luciFER//ZavrsniRad//Datasets//1//Clean1//" + c + ".txt", 
    "C://Users//Sugarplum//Desktop//luciFER//ZavrsniRad//Datasets//1//Results1//" + c + ".txt", 2000)"""


    #3. "3", "4", "5", "6", "7"
    #6. "3_22", "3_29", "4_5", "4_12", "4_19", "4_26", "5_3", "5_10", "5_17", "5_24", "5_31", 
    # "6_7", "6_14", "6_21", "6_28", "7_5"
    #1. "1_1", "1_2", "2_1", "2_2", "3_1", "3_2", "3_3_coronastart", "3_4", "4_1", "4_2", "4_3", "4_4", 
    # "5_1", "5_2", "5_3", "5_4", "6_1", "6_2", "6_3", "6_4", "7", "10", "12"

    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
