# SOULP.PY
Sentiment analysis app for classifying text into groups of negative, neutral or positive using embedded feature vectors.

![TreniranjeStablaOdluke](https://user-images.githubusercontent.com/110941477/222780545-525915a1-0cc1-466c-8a09-36180a973b23.png)
<p align="center">
Image 1: Decision tree diagram of activity <br />
</p>

This app is created through following steps: <br />
1. Create usable datasets consisting of preprocessed tweets and sentiment labels
2. Tokenize tweets <br />
3. Train a Word2Vec model with tokenized tweets <br />
4. Using tokenized tweets and Word2Vec model, create average vectors for each tweet <br />
5. Using average vectors and their coresponding sentiment labels, train the decision tree <br />
<br />

![positive](https://user-images.githubusercontent.com/110941477/222780594-4b78d671-9ed6-4903-bec3-744b377b3991.png)
<p align="center">
Image 2: User interface <br />
</p>
