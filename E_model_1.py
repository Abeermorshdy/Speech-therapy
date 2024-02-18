import os
import re
import nltk
import speech_recognition as sr
from gtts import gTTS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import pickle

nltk.download('stopwords')
nltk.download('punkt')

def iterate_english_sentences(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Split the content into sentences using regular expressions
        sentences = re.split(r'[.?!]', content)
        # Filter out empty sentences and return the sentences list
        return [sentence.strip() for sentence in sentences if sentence.strip()]


def read_sentence(S):
    counter = 1
    for sentence in S:
        sentences = sentence.split('\n')
        for s in sentences:
            file_name = f"text_{counter}.mp3"
            obj = gTTS(text=s,  lang='en', slow=False)
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
            t = r.recognize_google(audio, language='en-US')  # Recognize English speech
            transcriptions.append(t)

    return transcriptions

def calculate_average_similarity_english(file_path, folder_path):
    # Step 1: Read the sentences from the text file
    sentences = iterate_english_sentences(file_path)

    # Step 2: Convert the sentences to audio and prompt the user to repeat them
    read_sentence(sentences)

    # Step 3: Recognize the user's speech and get the transcriptions
    transcriptions = []
    for i in range(1, len(sentences) + 1):
        file_name = f"text_{i}.wav"
        transcription = STL(folder_path)
        transcriptions.extend(transcription)  # Extend transcriptions list

    # Step 4: Calculate the average similarity using BERT
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Load pre-trained BERT model
    
    # Check if transcriptions is empty
    if not transcriptions:
        print("No transcriptions found. Exiting.")
        return 0.0
    
    sentence_embeddings1 = model.encode(sentences, convert_to_tensor=True)
    transcriptions_embeddings = model.encode(transcriptions, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(sentence_embeddings1, transcriptions_embeddings).numpy().diagonal()
    average_similarity = sum(similarities) / len(similarities)

    return average_similarity


# Example usage
file_path = "English_sentences.txt"
folder_path = "English_records"
average_similarity = calculate_average_similarity_english(file_path, folder_path)
print("Average Similarity:", average_similarity)


# Save the model as PKL
with open('E_model.pkl', 'wb') as file:
    pickle.dump(calculate_average_similarity_english, file)