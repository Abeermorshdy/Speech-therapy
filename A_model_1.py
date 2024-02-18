import re
from gtts import gTTS
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pickle
import os

nltk.download('stopwords')
nltk.download('punkt')

def iterate_arabic_sentences(file_path):
    S = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Split the content into sentences using regular expressions
        sentences = re.split(r'[.?!]', content)
        # Filter out empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        # Return the sentences list
        return sentences

def read_sentence(S):
    counter = 1
    for sentence in S:
        sentences = sentence.split('\n')
        for s in sentences:
            file_name = f"text_{counter}.mp3"
            obj = gTTS(text=s, lang='ar', slow=False)
            obj.save(file_name)
            counter += 1

def STL(folder_path):
    r = sr.Recognizer()
    transcriptions = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            with sr.AudioFile(file_path) as source:
                audio = r.record(source)
            t = r.recognize_google(audio, language='ar-AR')
            transcriptions.append(t)

    return transcriptions

def calculate_cosine_similarity(X, Y):
    # Tokenization
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

    # Remove stopwords
    sw = stopwords.words('arabic')
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    # Form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    l1 = []
    l2 = []

    for w in rvector:
        if w in X_set:
            l1.append(1)
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)

    # Calculate cosine similarity
    c = 0
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1) * sum(l2))**0.5)
    return cosine

def calculate_average_similarity(X_list, Y_list):
    total_similarity = 0.0
    num_pairs = min(len(X_list), len(Y_list))

    for i in range(num_pairs):
        X = X_list[i]
        Y = Y_list[i]
        similarity = calculate_cosine_similarity(X, Y)
        total_similarity += similarity

    average_similarity = total_similarity / num_pairs
    return average_similarity

def calculate_average_similarity_arabic(file_path, folder_path):
    # Step 1: Read the sentences from the text file
    sentences = iterate_arabic_sentences(file_path)

    # Step 2: Convert the sentences to audio and prompt the user to repeat them
    read_sentence(sentences)

# Step 3: Recognize the user's speech and get the transcriptions
    transcriptions = []
    for i in range(1, len(sentences) + 1):
        file_name = f"text_{i}.mp3"
        #transcription = STL(folder_path)
        transcriptions = STL(folder_path)

    # Step 4: Calculate the average similarity
    average_similarity = calculate_average_similarity(sentences, transcriptions)

    return average_similarity

# Example usage
file_path = "Sentences.txt"
folder_path = "records"
average_similarity = calculate_average_similarity_arabic(file_path,folder_path)
print("Average Similarity:", average_similarity)