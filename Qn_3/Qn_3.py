

!pip install contractions

pip install gensim

pip install pydot

from textblob import Word
import nltk
nltk.download('wordnet')

pip install tensorflow

# Libraray import
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white",font_scale=1.5)
sns.set(rc={"axes.facecolor":"#FFFAF0","figure.facecolor":"#FFFAF0"})
sns.set_context("poster",font_scale = .7)

import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
import spacy
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
import re,string,unicodedata
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from scipy.sparse import lil_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import pos_tag

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv("/content/synthetic_news_data.csv")  # Change to your actual file path

# Display basic info
print("Full Dataset Shape:", df.shape)
print(df.head())

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)  # 80% train, 20% test

train_df.sample(5)

test_df.sample(5)

#### 3.1. Computing Dimension of Dataset
print("train_df shape: ",train_df.shape)
print("test_df shape: ",test_df.shape
      
### 3.2. Statistical Summary of Dataset

train_df.info()


#### 3.3. Checking if There's Any Duplicate Recordsprint("Duplicates in Dataset: ",train_df.duplicated().sum())


#### 3.4. Computing Total No. of Missing Values and the Percentage of Missing Values


missing_data = train_df.isnull().sum().to_frame().rename(columns={0:"Total No. of Missing Values"})
missing_data["% of Missing Values"] = round((missing_data["Total No. of Missing Values"]/len(train_df))*100,2)
missing_data


#### 3.5. Performing Descriptive Analysis

round(train_df.describe().T,2)

<center><div style='color:#ffffff;
           display:inline-block;
           padding: 5px 5px 5px 5px;
           border-radius:5px;
           background-color:#78D1E1;
           font-size:100%;'><a href=#toc style='text-decoration: none; color:#03001C;'>⬆️ Back To Top</a></div></center>

      
# 4 | Preprocessing

<div style="padding: 4px;color:white;margin:10;font-size:200%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://i.postimg.cc/fTDmwnkQ/Miaka.png); background-size: 100% auto;"></div>


### **1. Dropping Duplicates and Null Values**


train_df.dropna(inplace = True)


### **2. Lowercasing**


train_df['article'] = train_df['article'].str.lower()
train_df['summary'] = train_df['summary'].str.lower()
test_df['article'] = test_df['article'].str.lower()
test_df['summary'] = test_df['summary'].str.lower()


### **3. Removing contraction**


pip install contractions

import contractions

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

train_df['article'] = train_df['article'].apply(expand_contractions)
train_df['summary'] = train_df['summary'].apply(expand_contractions)
test_df['article'] = test_df['article'].apply(expand_contractions)
test_df['summary'] = test_df['summary'].apply(expand_contractions)

train_df['summary'] = ['<start> ' + sentence + ' <end>' for sentence in train_df['summary']]
test_df['summary'] = ['<start> ' + sentence + ' <end>' for sentence in test_df['summary']]


### **4. Tokenize on the Data**

from tensorflow.keras.preprocessing.text import Tokenizer

tok = Tokenizer()
tok.fit_on_texts(train_df['article']+train_df['summary'])

len(tok.word_index)

tok.document_count

# # Ensure all entries are strings
# train_df['article'] = train_df['article'].astype(str)
# train_df['summary'] = train_df['summary'].astype(str)
# test_df['article'] = test_df['article'].astype(str)
# test_df['summary'] = test_df['summary'].astype(str)

# Now convert to sequences
train_df['article'] = tok.texts_to_sequences(train_df['article'])
train_df['summary'] = tok.texts_to_sequences(train_df['summary'])

test_df['article'] = tok.texts_to_sequences(test_df['article'])
test_df['summary'] = tok.texts_to_sequences(test_df['summary'])

train_df

def calculate_max_sequence_length(train_summary,train_article,test_summary,test_article):
    max_length_1 = max(max(len(seq) for seq in train_article), max(len(seq) for seq in train_summary))
    max_length_2 = max(max(len(seq) for seq in test_article), max(len(seq) for seq in test_summary))
    max_length_combined = max(max_length_1,max_length_2)
    return max_length_combined

max_length_combined = calculate_max_sequence_length(train_df['article'],train_df['summary'],test_df['article'],test_df['summary'])
max_length_combined

for num in train_df['article'][1]:
    print(num, end=' ')

# Print corresponding words horizontally
for num in train_df['article'][1]:
    word = tok.index_word.get(num, 'UNK')
    print(word, end=' ')

for num in train_df['summary'][1]:
    word = tok.index_word.get(num)
    print(word,end = " ")


### **10. separating the data in dependent and independent and padding it**


x_train = pad_sequences(train_df['article'],maxlen = max_length_combined,padding = 'post')
y_train = pad_sequences(train_df['summary'],maxlen = max_length_combined,padding = 'post')
x_test = pad_sequences(test_df['article'],maxlen = max_length_combined,padding = 'post')
y_test = pad_sequences(test_df['summary'],maxlen = max_length_combined,padding = 'post')

x_train

y_train

# Check the shape of the resulting arrays
print("Shape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


### **5.1. ENCODER-DECODER MODEL**


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Define dimensions and vocab sizes
max_length_input = x_train.shape[1]
max_length_output = y_train.shape[1]
input_vocab_size = len(tok.word_index) + 1
output_vocab_size = len(tok.word_index) + 1

# Define Encoder
encoder_inputs = Input(shape=(max_length_input,))
encoder_embedding = Embedding(input_dim=input_vocab_size, output_dim=260)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(64, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Define Decoder
decoder_inputs = Input(shape=(max_length_output,))
decoder_embedding = Embedding(input_dim=output_vocab_size, output_dim=260)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Dense layer to generate final output
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Full model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

from tensorflow.keras.utils import plot_model
from IPython.display import Image
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
Image('model_plot.png')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('model_checkpoint.keras', save_best_only=True)  # Updated filepath

# Train the model with callbacks
history = model.fit(
    x=[x_train, y_train],
    y=y_train,
    batch_size=8,
    epochs=5,
    validation_data=([x_test, y_test], y_test),
    callbacks=[early_stopping, model_checkpoint],
)


### **5.2. ENCODER-DECODER MODEL **


epochs_range = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'], label='Training Loss', marker='o')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs_range)
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epochs_range)
plt.legend()

plt.tight_layout()
plt.show()

import nltk
nltk.download('punkt')

from nltk.translate.bleu_score import corpus_bleu
import numpy as np

# Reverse token dictionary for summary (assuming you have a tokenizer `tok` for summary)
rev_tok_summary = {idx: word for word, idx in tok.word_index.items()}

# Define the batch size
batch_size = 5

# Initialize lists to store predictions and references
predicted_summaries = []
references = []

# Predict on x_test data in batches
for start in range(0, len(x_test), batch_size):
    end = min(start + batch_size, len(x_test))
    x_batch = x_test[start:end]
    y_batch = y_test[start:end]  # Use the corresponding y_test for padding purposes

    # Predict on the batch
    predictions = model.predict([x_batch, y_batch], batch_size=batch_size)

    # Convert predicted tokens to sentences (predicted summaries)
    predicted_tokens_np = np.argmax(predictions, axis=-1)

    for sample in predicted_tokens_np:
        # Convert the predicted tokens into the corresponding words for the summary
        predicted_sentence = ' '.join([rev_tok_summary.get(token, '<unknown>') for token in sample if token != 0 and token not in [tok.word_index.get('start'), tok.word_index.get('end')]])
        predicted_summaries.append(predicted_sentence)

    # Extract the true (actual) summaries for the BLEU score references
    for i in range(len(y_batch)):
        true_summary_sentence = ' '.join([rev_tok_summary.get(token, '<unknown>') for token in y_batch[i] if token != 0 and token not in [tok.word_index.get('start'), tok.word_index.get('end')]])
        references.append([true_summary_sentence.split()])  # BLEU expects list of references for each hypothesis

# Calculate BLEU score
bleu_score = corpus_bleu(references, predicted_summaries)
print("BLEU Score: ", bleu_score)


### **5.4. Saving the best weight**


# Save the entire model
model.save('encoder_decoder_model.h5')
print('Model saved succesfully!!')

