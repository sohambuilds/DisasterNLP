import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('disaster_tweet_classifier.h5')


with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

def classify_tweet(text):
    
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    
    prediction = model.predict(padded_sequences)[0]

    # Classify based on threshold (adjust as needed)
    threshold = 0.5
    if prediction > threshold:
        classification = "Disaster"
    else:
        classification = "Not Disaster"

    return classification

# Example usage
tweet_text = "Earthquake in Tokyo!"
classification = classify_tweet(tweet_text)
print(classification)
