import pandas as pd
import numpy as np
import os
import re
import io
import matplotlib.pyplot as plt
import spacy
import scispacy

from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score, accuracy_score, \
					confusion_matrix, precision_score, recall_score,  \
					f1_score, roc_curve, auc, classification_report, \
						plot_roc_curve

import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
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
    return text

dfUadm['TEXT_CONCAT'] = dfUadm['TEXT_CONCAT'].apply(final_clean)

X = dfUadm.TEXT_CONCAT
Y = dfUadm.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=77, stratify=Y)

"""
#1. make tf.dataset, consider tf.keras.preprocessing.text_dataset_from_dictionary

batch_size = 512
seed = 123

train_ds = tf.data.Dataset.from_tensor_slices(x_train)
val_ds = tf.data.Dataset.from_tensor_slices(x_train.sample)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
"""
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

tf.config.set_visible_devices([], 'GPU')

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_md")
# TextVectorization - to work on different lengths 
# vectorize top 0.5M words
# each sequence padded to 2000 tokens TODO: find out doc length
vocab_size = 1000
maxlen = 20

vectorizer = TextVectorization(
					max_tokens=vocab_size, 
					standardize=None,
					output_sequence_length=maxlen,
					output_mode='int')

# text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128)
vectorizer.adapt(x_train[0:10].to_numpy())

# get vocabulary ie. vector mappings of each word
vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))

# test vectorizer (optional)
vectorizer.get_vocabulary()[:10]
t1 = vectorizer([['medical discharge coronary artery disease']])
t1.numpy()[0, :5]

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


model = tf.keras.Sequential([
	vectorizer,
    embedding_layer,
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=[tf.keras.metrics.AUC()])

history = model.fit(x_train[0:10].to_numpy(), y_train, epochs=1, batch_size=5,
                    validation_split=0.15)


history = model.fit(x_train[0:10].to_numpy(), y_train[0:10].to_numpy(), epochs=1)
model.summary()

def evaluation_plotting(history, model, save_path):

	y_pred = model.predict(x_test)
	y_pred_classes = (model.predict(x_test) > 0.5).astype("int32")

	accuracy = accuracy_score(y_test, y_pred_classes)
	precision = precision_score(y_test, y_pred_classes)
	recall = recall_score(y_test, y_pred_classes)
	f1 = f1_score(y_test, y_pred_classes)
	auc = roc_auc_score(y_test, y_pred)
	matrix = confusion_matrix(y_test, y_pred_classes)

	print('Accuracy: %f' % accuracy)
	print('Precision: %f' % precision)
	print('Recall: %f' % recall)
	print('F1 score: %f' % f1)
	print('ROC AUC: %f' % auc)
	print(matrix)

	history_df = pd.DataFrame(history.history)
	history_df.to_csv(os.path.join(work_dir, 'Output/windows', save_path, \
				"{}_history.csv".format(save_path)), index=False)
	model.save(os.path.join(work_dir, 'Output/windows', save_path))
	np.savetxt(os.path.join(work_dir, 'Output/windows', save_path, \
				"{}_ypred.csv".format(save_path)), y_pred, delimiter=",")
	y_test.to_csv(os.path.join(work_dir, 'Output/windows', save_path, \
				"{}_ytest.csv".format(save_path)), index=True)

	# plot loss, accuracy and roc curves
	plot_loss(history, save_path)
	plot_accuracy(history, save_path)
	plot_roc(history, y_pred, save_path)

	return y_pred


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def plot_loss(history, save_path):
	plt.plot(history.history['loss'], label='loss')
	plt.plot(history.history['val_loss'], label='val_loss')
	plt.title('Binary Crossentropy Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend(['train', 'val'], loc='best')
	plt.grid(True)
	plt.savefig(os.path.join(work_dir, 'Output/windows', save_path, "{}_loss.png".format(save_path)))
	plt.close()

def plot_accuracy(history, save_path):
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Accuracy Over Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(['train', 'val'], loc='best')
	plt.grid(True)
	plt.savefig(os.path.join(work_dir, 'Output/windows', save_path, "{}_accuracy.png".format(save_path)))
	plt.close()

def plot_roc(history, y_pred, save_path):

	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	auc_keras = auc(fpr, tpr)

	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_keras))

	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend('lower right')
	plt.savefig(os.path.join(work_dir, 'Output/windows', save_path, "{}_roc.png".format(save_path)))
	plt.close()

"""
## run functions in for loop : train 4 windows at once ##

for n in range(len(window_list)):
	print(n, " - windowing set {}: ".format(window_paths[n]), datetime.now())
	tempTime = datetime.now()

	model, x_train, x_test, y_train, y_test, STEPS_PER_EPOCH = windowed_model(dfuk, window_list[n])
	history, model = train_model(model, x_train, y_train)
	y_pred_l2 = evaluation_plotting(history, model, window_paths[n])

	print("Duration of {}-window training: ".format(window_paths[n]), datetime.now() - tempTime)
	clear_session()
"""

"""

def show_metrics(pred_tag, y_test):
    print("F1-score: ", f1_score(pred_tag, y_test))
    print("Precision: ", precision_score(pred_tag, y_test))
    print("Recall: ", recall_score(pred_tag, y_test))
    print("Acuracy: ", accuracy_score(pred_tag, y_test))
    print("-"*50)
    print(classification_report(pred_tag, y_test))
    

def evaluation_plotting(model):

	y_pred_classes = model.predict(x_test_dtm)
	y_pred = (model.predict_proba(x_test_dtm))[:, 1]

	accuracy = accuracy_score(y_test, y_pred_classes)
	precision = precision_score(y_test, y_pred_classes)
	recall = recall_score(y_test, y_pred_classes)
	f1 = f1_score(y_test, y_pred_classes)
	auc = roc_auc_score(y_test, y_pred)
	matrix = confusion_matrix(y_test, y_pred_classes)

	print('Accuracy: %f' % accuracy)
	print('Precision: %f' % precision)
	print('Recall: %f' % recall)
	print('F1 score: %f' % f1)
	print('ROC AUC: %f' % auc)
	print(matrix)

	plot_roc(y_pred)

	return y_pred

def plot_roc(y_pred):

	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	auc_logreg = auc(fpr, tpr)

	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Logistic Regression (area = {:.3f})'.format(auc_logreg))

	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.savefig(os.path.join(work_dir, "Models/Logreg/roc.png"))
	plt.show()



"""