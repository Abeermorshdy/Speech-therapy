# Speech-therapy
This repo is an implementation for Arabic and English language speech trainer for babies from 2 years to 6 years To help them get rid of stuttering

## Requirements

- Python 3.x
- Dependencies: gtts, speechrecognition, nltk,sentence_transformers
- You can study the code to understand from this file `main.ipynb`

Install the dependencies using pip

## Usage

1. Prepare your input text file containing the Arabic or English sentences you want to compare. Each sentence should be on a new line, check `English_sentences.txt` file. 
2. Prepare the input records in a folder called `English_records` or `Arabic_records` contains the child's voice.

3. Download `E_model.pkl` from here [link](https://drive.google.com/file/d/1N6o4c8WMsxlNPTURt8RNKy5q6cdtUskE/view?usp=drive_link) and Import the model and load it from the PKL file:

```import pickle
# Load the model from the PKL file
with open('similarity_calculator.pkl', 'rb') as file:
    model = pickle.load(file)
file_path = "path/to/your/text/file.txt"
folder_path = "path/to/your/folder"
average_similarity = model(file_path,folder_path)
