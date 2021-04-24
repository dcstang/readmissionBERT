import pandas as pd
import numpy as np
import os
import re
import io

import spacy
import scispacy
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

path = os.getcwd()
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project')

dfUadm = pd.read_csv(os.path.join(work_dir, 'Data/lemma_dfUadm.csv'),
	  dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
		 'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
		 'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
		 'AGE' : 'UInt8'},
	  parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
	  header=0)

def final_clean(text):
	text = re.sub(r"[^a-zA-Z\d\s:]", "", text)
	text = re.sub(r"\b[a-zA-Z]\b", "", text)
	text = re.sub(r"\b\w{1,4}\b", "", text)
	text = re.sub(r"\s\s+", " ", text)
	text = re.sub(r"(^(?:\S+\s+\n?){1,20})", "", text)
	return text

dfUadm['TEXT_CONCAT'] = dfUadm['TEXT_CONCAT'].apply(final_clean)

# train-test-split
X = dfUadm.TEXT_CONCAT
Y = dfUadm.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=77, stratify=Y)

# embeddings for logreg
nlp = spacy.load("en_core_sci_md")

vocab_size = 60000
maxlen = 100

vectorizer = TextVectorization(
					max_tokens=vocab_size, 
					standardize=None,
					output_sequence_length=maxlen,
					output_mode='int')

vectorizer.adapt(x_train[0:25000].to_numpy())

# get vocabulary ie. vector mappings of each word
vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))

#generate the embedding matrix
embedding_dim = len(nlp('The').vector)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for i, word in enumerate(vocab):
	embedding_matrix[i] = nlp(word).vector

print("Found %s word vectors." % len(embedding_matrix))


# load pre-trained embeddings into embedding layer
embedding_layer = Embedding(
	vocab_size,
	embedding_dim,
	embeddings_initializer=keras.initializers.Constant(embedding_matrix),
	trainable=False, 
	mask_zero=True
)

text_to_embeddings = tf.keras.Sequential([
	vectorizer,
	embedding_layer
])

weights = text_to_embeddings.get_layer('embedding_1').get_weights()[0]

out_v = io.open(os.path.join(work_dir, 'Models/Embeddings/vectors.tsv'), 'w', encoding='utf-8')
out_m = io.open(os.path.join(work_dir, 'Models/Embeddings/metadata.tsv'), 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()

out_v = pd.read_csv(os.path.join(work_dir, 'Models/Embeddings/vectors.tsv'), sep='\t', header=None)
out_m = pd.read_csv(os.path.join(work_dir, 'Models/Embeddings/metadata.tsv'), header=None)

(out_v.sum(axis=1) == 0).sum()
out_m = out_m[(out_v.sum(axis=1) != 0)]
out_v = out_v[(out_v.sum(axis=1) != 0)]

out_m.to_csv(os.path.join(work_dir, 'Models/Embeddings/vectors_cleaned.tsv'), index=False, sep='\t', header=False)
out_v.to_csv(os.path.join(work_dir, 'Models/Embeddings/metadata_cleaned.tsv'), index=False, sep='\t', header=False)