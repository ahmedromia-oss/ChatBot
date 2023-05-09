import nltk
import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from nltk.corpus import treebank, brown, conll2000
from sklearn.model_selection import train_test_split
from tensorflow import keras

nltk.download('treebank')
nltk.download('brown')
nltk.download('conll2000')
nltk.download('universal_tagset')
# Download all PoS-tagged sentences and place them in one list.
tagged_sentences = treebank.tagged_sents(tagset='universal') +\
                   brown.tagged_sents(tagset='universal') +\
                   conll2000.tagged_sents(tagset='universal')

print(tagged_sentences[0])
print(f"Dataset size: {len(tagged_sentences)}")

sentences, sentence_tags = [], []

for s in tagged_sentences:
  sentence, tags = zip(*s)
  sentences.append(list(sentence))
  sentence_tags.append(list(tags))
  
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10
x_train, x_test, y_train, y_test = train_test_split(sentences, sentence_tags, 
                                                    test_size=1 - train_ratio, 
                                                    random_state=1)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                test_size=test_ratio/(test_ratio + validation_ratio), 
                                                random_state=1)
sentence_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='<OOV>')

sentence_tokenizer.fit_on_texts(x_train)
tag_tokenizer = keras.preprocessing.text.Tokenizer()
tag_tokenizer.fit_on_texts(y_train)
tag_tokenizer.get_config()
tag_tokenizer.word_index
x_train_seqs = sentence_tokenizer.texts_to_sequences(x_train)
y_train_seqs = tag_tokenizer.texts_to_sequences(y_train)
tag_tokenizer.sequences_to_texts([y_train_seqs[0]])
x_val_seqs = sentence_tokenizer.texts_to_sequences(x_val)
y_val_seqs = tag_tokenizer.texts_to_sequences(y_val)
MAX_LENGTH = len(max(x_train_seqs, key=len))
x_train_padded = keras.preprocessing.sequence.pad_sequences(x_train_seqs, padding='post', 
                                                            maxlen=MAX_LENGTH)
y_train_padded = keras.preprocessing.sequence.pad_sequences(y_train_seqs, padding='post', 
                                                            maxlen=MAX_LENGTH)
x_val_padded = keras.preprocessing.sequence.pad_sequences(x_val_seqs, padding='post', maxlen=MAX_LENGTH)
y_val_padded = keras.preprocessing.sequence.pad_sequences(y_val_seqs, padding='post', maxlen=MAX_LENGTH)
y_train_categoricals = keras.utils.to_categorical(y_train_padded)
idx = np.argmax(y_train_categoricals[0][0])
y_val_categoricals = keras.utils.to_categorical(y_val_padded)
num_tokens = len(sentence_tokenizer.word_index) + 1
print(num_tokens)
embedding_dim = 128

# For the output layer. The number of classes corresponds to the 
# number of possible tags.
num_classes = len(tag_tokenizer.word_index) + 1
tf.random.set_seed(0)

model = keras.Sequential()

model.add(layers.Embedding(input_dim=num_tokens, 
                           output_dim=embedding_dim, 
                           input_length=MAX_LENGTH,
                           mask_zero=True))

model.add(layers.Bidirectional(layers.LSTM(10, return_sequences=True, 
                                           kernel_initializer=tf.keras.initializers.random_normal(seed=1))))

model.add(layers.Dense(num_classes, activation='softmax', 
                       kernel_initializer=tf.keras.initializers.random_normal(seed=1)))

model.compile(loss='categorical_crossentropy',
            #   optimizer='adam',
              metrics=['accuracy'])
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


history = model.fit(x_train_padded, y_train_categoricals, epochs=1, 
                    batch_size=2028, validation_data=(x_val_padded, y_val_categoricals), 
                    callbacks=[es_callback])

x_test_seqs = sentence_tokenizer.texts_to_sequences(x_test)
x_test_padded = keras.preprocessing.sequence.pad_sequences(x_test_seqs, padding='post', maxlen=MAX_LENGTH)

y_test_seqs = tag_tokenizer.texts_to_sequences(y_test)
y_test_padded = keras.preprocessing.sequence.pad_sequences(y_test_seqs, padding='post', maxlen=MAX_LENGTH)
y_test_categoricals = keras.utils.to_categorical(y_test_padded)
samples = [
    "Brown refused to testify.",
    "Brown sofas are on sale.",
]
def tag_sentences(sentences):
  sentences_seqs = sentence_tokenizer.texts_to_sequences(sentences)
  sentences_padded = keras.preprocessing.sequence.pad_sequences(sentences_seqs, 
                                                                maxlen=MAX_LENGTH, 
                                                                padding='post')

  # The model returns a LIST of PROBABILITY DISTRIBUTIONS (due to the softmax)
  # for EACH sentence. There is one probability distribution for each PoS tag.
  tag_preds = model.predict(sentences_padded)

  sentence_tags = []

  # For EACH LIST of probability distributions...
  for i, preds in enumerate(tag_preds):

    # Extract the most probable tag from EACH probability distribution.
    # Note how we're extracting tags for only the non-padding tokens.
    tags_seq = [np.argmax(p) for p in preds[:len(sentences_seqs[i])]]

    # Convert the sentence and tag sequences back to their token counterparts.
    words = [sentence_tokenizer.index_word[w] for w in sentences_seqs[i]]
    tags = [tag_tokenizer.index_word[t] for t in tags_seq]
    sentence_tags.append(list(zip(words, tags)))

  return sentence_tags

tagged_sample_sentences = tag_sentences(samples)
print(tagged_sample_sentences[0])
