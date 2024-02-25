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

class SimilarityCalculator:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def iterate_english_sentences(self, file_path):
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Split the content into sentences using regular expressions
            sentences = re.split(r'[.?!]', content)
            # Filter out empty sentences and return the sentences list
            return [sentence.strip() for sentence in sentences if sentence.strip()]

    def read_sentence(self, S):
        counter = 1
        for sentence in S:
            sentences = sentence.split('\n')
            for s in sentences:
                file_name = f"text_{counter}.mp3"
                obj = gTTS(text=s,  lang='en', slow=False)
                obj.save(file_name)
                counter += 1

    def STL(self, folder_path):
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

    def calculate_average_similarity_english(self, file_path, folder_path):
        # Step 1: Read the sentences from the text file
        sentences = self.iterate_english_sentences(file_path)

        # Step 2: Convert the sentences to audio and prompt the user to repeat them
        self.read_sentence(sentences)

        # Step 3: Recognize the user's speech and get the transcriptions
        transcriptions = []
        for i in range(1, len(sentences) + 1):
            file_name = f"text_{i}.wav"
            transcription = self.STL(folder_path)
            transcriptions.extend(transcription)  # Extend transcriptions list

        # Step 4: Calculate the average similarity using BERT
        if not transcriptions:
            print("No transcriptions found. Exiting.")
            return 0.0

        sentence_embeddings1 = self.model.encode(sentences, convert_to_tensor=True)
        transcriptions_embeddings = self.model.encode(transcriptions, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(sentence_embeddings1, transcriptions_embeddings).numpy().diagonal()
        average_similarity = sum(similarities) / len(similarities)

        return average_similarity

# Instantiate the class
similarity_calculator = SimilarityCalculator()

# Save the class instance to a .pkl file
with open('similarity_calculator.pkl', 'wb') as file:
    pickle.dump(similarity_calculator, file)
