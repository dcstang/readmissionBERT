import pandas as pd
import numpy as np
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import re
import io
import matplotlib.pyplot as plt
import spacy
import scispacy
from textaugment import EDA

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

from datetime import datetime

# time checkpoint
startTime = datetime.now()
print("Init time: ", startTime)

# set intra_op to number of physical cores, experiement with inter_op
tf.config.threading.set_intra_op_parallelism_threads(20)
tf.config.threading.set_inter_op_parallelism_threads(15)
print(tf.config.threading.get_inter_op_parallelism_threads())

work_dir = os.getcwd() #.. mimic/script

dfUadm = pd.read_csv(os.path.join(work_dir, '../lemmatized/lemma_dfUadm.csv'),
	  dtype={'SUBJECT_ID' : 'UInt32', 'HADM_ID' : 'UInt32', 'TEXT_CONCAT' : 'string',
		 'ADMISSION_TYPE' : 'string', 'ETHNICITY' : 'string', 'DIAGNOSIS' : 'string',
		 'HOSPITAL_EXPIRE_FLAG' : 'bool', 'MORTALITY_30D' : 'bool', 'TARGET' : 'bool',
		 'AGE' : 'UInt8'},
	  parse_dates=['ADMITTIME', 'DISCHTIME', 'NEXT_UADMITTIME', 'DOD_SSN'],
	  header=0)

def final_clean(text):
	text = re.sub(r"[^a-zA-Z\d\s:]", "", text)
	text = re.sub(r"\b[a-zA-Z]\b", "", text)
	text = re.sub(r"\s\s+", " ", text)
	text = re.sub(r"(^(?:\S+\s+\n?){1,12})", "", text)
	return text

dfUadm['TEXT_CONCAT'] = dfUadm['TEXT_CONCAT'].apply(final_clean)

X = dfUadm.TEXT_CONCAT
Y = dfUadm.TARGET

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state=77, stratify=Y)
del dfUadm # save some memory

# perform text augmentation of x_train
print("Initial stratified test_train_split 0.15 and info about train_set:")
print("Number and prop(%) of cases   : ", (y_train == 1).sum(), 
			", % =", round((y_train == 1).sum()/len(y_train), 3))
print("Number and prop(%) of controls: ", (y_train == 0).sum(), 
			", % =", round((y_train == 0).sum()/len(y_train), 3))

repeatCases = len(y_train.index[y_train == True])
x_train = x_train.append([x_train[y_train[y_train==True].index]]*4)
y_train = y_train.append([y_train[y_train==True]]*4)

# text augmentation to replicated true cases
t = EDA()

x_train[-repeatCases*2:-repeatCases] = x_train[-repeatCases*2:-repeatCases].apply(t.random_deletion, args=(0.15,))
x_train[-repeatCases*3:-repeatCases*2] = x_train[-repeatCases*3:-repeatCases*2].apply(t.random_swap, args=(150,))
x_train[-repeatCases*4:-repeatCases*3] = x_train[-repeatCases*4:-repeatCases*3].apply(t.random_deletion, args=(0.25,))

print("Post text augmentation train_set:")
print("Augmentation - random deletion 15-25%, random swap 150 words")
print("Number and prop(%) of cases   : ", (y_train == 1).sum(), 
			", % =", round((y_train == 1).sum()/len(y_train), 3))
print("Number and prop(%) of controls: ", (y_train == 0).sum(), 
			", % =", round((y_train == 0).sum()/len(y_train), 3))

print("x_train dims: ", x_train.shape)
print("x_test dims : ", x_test.shape, "\n")

nlp = spacy.load("en_core_sci_md")

# TextVectorization layer settings:
vocab_size = 50000
maxlen = 400

vectorizer = TextVectorization(
					max_tokens=vocab_size, 
					standardize=None,
					output_sequence_length=maxlen,
					output_mode='int')

vectorizer.adapt(x_train[100:30000].to_numpy())

# get vocabulary ie. vector mappings of each word
vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))

#generate the embedding matrix
embedding_dim = len(nlp('The').vector)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for i, word in enumerate(vocab):
	embedding_matrix[i] = nlp(word).vector

print("Found %s word vectors." % len(embedding_matrix))

# LSTM Settings
MAX_EPOCHS = 10
BATCHSIZE = 16
LEARNING_RATE = 0.001
opt_key = 'ADAM'
LSTM_FIRST_NEURONS = 16
BIDIRECTIONAL = True

# load pre-trained embeddings into embedding layer
embedding_layer = Embedding(
	vocab_size,
	embedding_dim,
	embeddings_initializer=keras.initializers.Constant(embedding_matrix),
	trainable=False, 
	mask_zero=True
)

if BIDRECTIONAL == True:
	# bidirectional implementation
	model = tf.keras.Sequential([
		vectorizer,
		embedding_layer,
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_FIRST_NEURONS)),
		tf.keras.layers.Dense(16, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')
	])
else:
	model = tf.keras.Sequential([
		vectorizer,
		embedding_layer,
		tf.keras.layers.LSTM(LSTM_FIRST_NEURONS),
		tf.keras.layers.Dense(16, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')
	])

def get_optimizer(opt_key, LEARNING_RATE=0.001):
	if opt_key == 'ADAM':
		optimizer_selected = tf.keras.optimizers.Adam(LEARNING_RATE)
	elif opt_key == 'SGD':
		optimizer_selected = tf.keras.optimizers.SGD(lr=LEARNING_RATE)
	else:
		optimizer_selected = tf.keras.optimizers.RMSprop(lr=LEARNING_RATE)
	return optimizer_selected

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
			  optimizer=get_optimizer(opt_key, LEARNING_RATE),
			  metrics=[tf.keras.metrics.AUC()])

"""
history = model.fit(x_train.to_numpy(), y_train.to_numpy(), epochs=1, batch_size=5,
					validation_split=0.15)
"""

# print settings before starting training for PBS records
print('======== EMBEDDINGS SETTINGS ==========')
print(f'Max vocabulary size {vocab_size}')
print(f'Max number of words per document {maxlen}')
print('======== LSTM SETTINGS ==========')
print(f'MAX_EPOCHS {MAX_EPOCHS}')
print(f'BATCH SIZE {BATCHSIZE}')
print(f'LEARNING RATE {LEARNING_RATE}')
print(f'OPTIMIZER {opt_key}')
print(f'FIRST LAYER LSTM n NEURONS {LSTM_FIRST_NEURONS}')
print(f'BIDIRECTIONAL {BIDIRECTIONAL}')

history = model.fit(x_train.to_numpy(), y_train.to_numpy(), 
			epochs=MAX_EPOCHS, batch_size=BATCHSIZE, verbose=2,
			validation_split=0.15)
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
	history_df.to_csv(os.path.join(work_dir, '../Models/LSTM', \
				"{}_history.csv".format(save_path)), index=False)
	np.savetxt(os.path.join(work_dir, '../Models/LSTM', \
				"{}_y_pred.csv".format(save_path)), y_pred, delimiter=",")
	y_test.to_csv(os.path.join(work_dir, '../Models/LSTM', \
				"{}_y_test.csv".format(save_path)), index=True)

	# plot loss, accuracy and roc curves
	plot_loss(history, save_path)
	plot_accuracy(history, save_path)
	plot_roc(history, y_pred, save_path)

	os.mkdir(os.path.join(work_dir, '../Models/LSTM', save_path))
	model.save(os.path.join(work_dir, '../Models/LSTM', save_path))
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
	plt.savefig(os.path.join(work_dir, '../Models/LSTM', "{}_loss.png".format(save_path)))
	plt.close()

def plot_accuracy(history, save_path):
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Accuracy Over Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(['train', 'val'], loc='best')
	plt.grid(True)
	plt.savefig(os.path.join(work_dir, '../Models/LSTM', "{}_accuracy.png".format(save_path)))
	plt.close()

def plot_roc(history, y_pred, save_path):

	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	auc_keras = auc(fpr, tpr)

	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='LSTM (area = {:.3f})'.format(auc_keras))

	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend('lower right')
	plt.savefig(os.path.join(work_dir, '../Models/LSTM', "{}_roc.png".format(save_path)))
	plt.close()


y_pred = evaluation_plotting(history, model, 'lstm_cpu_{}_{}_{}'.format(LSTM_FIRST_NEURONS, opt_key, BIDIRECTIONAL))

print("Batch job ended: ", datetime.now() - startTime)

"""
# training hyperparameters for NN
MAX_EPOCHS = 300
N_BATCH_SIZE = 32
STEPS_PER_EPOCH = len(x_train) // N_BATCH_SIZE
LEARNING_RATE = 0.001

model_names = [model_01_bn2hl2, model_02_1hbnl2, model_03_bn1hl2, model_04_bn1hl2, \
				model_05_bn2hl2, model_06_bn2hl2, model_07_bn1hdo, model_08_1hbndo]
model_paths = ['model_01_bn2hl2', 'model_02_1hbnl2', 'model_03_bn1hl2', 'model_04_bn1hl2', \
				'model_05_bn2hl2', 'model_06_bn2hl2', 'model_07_bn1hdo', 'model_08_1hbndo']

# early stopping callback  
early_stopping = tf.keras.callbacks.EarlyStopping(
						monitor='val_loss',
						mode='min', patience=15,
						min_delta=0.005) 

## pipeline of model compile > train > eval > plotting
def get_optimizer():
	return tf.keras.optimizers.SGD(lr=LEARNING_RATE) 

def build_and_compile(model, optimizer=None):

		if optimizer is None:
				optimizer=get_optimizer()

		model.compile(
				loss=keras.losses.BinaryCrossentropy(),
				optimizer=optimizer,
				metrics=[tf.keras.metrics.Accuracy(),
						keras.metrics.AUC(),
						keras.metrics.Precision(),
						keras.metrics.Recall()
				]
		)

		return model

def train_model(model):
	
	dnn_model = build_and_compile(model)

	history = dnn_model.fit(
			x_train, y_train,
			validation_split = 0.15,
			batch_size=N_BATCH_SIZE,
			verbose=2,
			epochs=MAX_EPOCHS,
			callbacks=[early_stopping]
	)

	dnn_model.summary()
	
	return history, dnn_model


for n in range(len(model_names)):
	print(n, " - Training model {}: ".format(model_paths[n]), datetime.now())
	tempTime = datetime.now()

	history, model = train_model(model_names[n])
	y_pred_l2 = evaluation_plotting(history, model, model_paths[n])

	print("Duration of {} model training: ".format(model_names[n]), datetime.now() - tempTime)
	clear_session() 


print("Experiment ended: ", datetime.now() - startTime)

"""