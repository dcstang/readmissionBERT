import pandas as pd
import numpy as np
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
import re
import sys
import matplotlib.pyplot as plt
import spacy
import scispacy

import tensorflow as tf
import tensorflow_text as text
import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.client import device_lib

from datetime import datetime


from sklearn.metrics import roc_auc_score, accuracy_score, \
					confusion_matrix, precision_score, recall_score,  \
					f1_score, roc_curve, auc, classification_report, \
					plot_roc_curve
# tf.data settings
BUFFER_SIZE = 48880
VALIDATION_RATIO = 0.1 

# TextVectorization layer settings:
vocab_size = 23000
maxlen = 150

# LSTM Settings
MAX_EPOCHS = 5
BATCHSIZE = 16
LEARNING_RATE = 0.0008
opt_key = 'ADAM'
LSTM_FIRST_NEURONS = 16
BIDIRECTIONAL = True


# time checkpoint
startTime = datetime.now()
print("Init time: ", startTime)

# check for CPU / GPUs
device_lib.list_local_devices() 

device_name = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
if device_name[0] == "/device:GPU:0":
    #device_name = "/gpu:0"
    print('GPU')
else:
    print('CPU')
    #device_name = "/cpu:0"

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# edit section above for HPC

path = os.getcwd() #.. mimic/script
work_dir = os.path.join(path, 'Sem 2 - Machine Learning/Project/Scripts')


TRAIN_FILE_NAMES = ['0_train_controls.txt', '1_train_cases.txt']
TEST_FILE_NAMES = ['0_test_controls.txt', '1_test_cases.txt']
DATA_DIR = os.path.join(work_dir, '../Data/LSTM_data')

def labeler(example, index):
	return example, tf.cast(index, tf.int32)


# tf.data ETL
labeled_train_sets = []

for idx, file_name in enumerate(TRAIN_FILE_NAMES):
	lines_dataset = tf.data.TextLineDataset(os.path.join(DATA_DIR, 'train', file_name))
	labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, idx))
	labeled_train_sets.append(labeled_dataset)

train_ds = labeled_train_sets[0]
train_ds = train_ds.concatenate(labeled_train_sets[1]) # add cases
train_ds = train_ds.shuffle(BUFFER_SIZE)

final_train_ds =  train_ds.skip(int(BUFFER_SIZE * VALIDATION_RATIO))
val_ds =  train_ds.take(int(BUFFER_SIZE * VALIDATION_RATIO))

final_train_ds =  train_ds.batch(BATCHSIZE)
final_train_ds.prefetch(1)
val_ds =  train_ds.batch(BATCHSIZE)


# test_set
labeled_test_sets = []

for idx, file_name in enumerate(TEST_FILE_NAMES):
	lines_dataset = tf.data.TextLineDataset(os.path.join(DATA_DIR, 'test', file_name))
	labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, idx))
	labeled_test_sets.append(labeled_dataset)

test_ds = labeled_test_sets[0]
test_ds = test_ds.concatenate(labeled_test_sets[1]) # add cases
test_ds = test_ds.batch(BATCHSIZE)

nlp = spacy.load("en_core_sci_md")
vectorizer = TextVectorization(
					max_tokens=vocab_size, 
					standardize=None,
					output_sequence_length=maxlen,
					output_mode='int')

# only on train_ds, no data leakage
train_text = final_train_ds.map(lambda text, labels: text)
vectorizer.adapt(train_text)
del train_text

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

if BIDIRECTIONAL == True:
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
			  metrics=[
				  	tf.keras.metrics.AUC(),
			  		tf.keras.metrics.Accuracy(),
					tf.keras.metrics.Precision(),
					tf.keras.metrics.Recall(),
					tf.keras.metrics.TrueNegatives(),
					tf.keras.metrics.TruePositives(),
					tf.keras.metrics.FalseNegatives(),
					tf.keras.metrics.FalsePositives()])


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
print('\n')

history = model.fit(
				final_train_ds, 
				epochs=MAX_EPOCHS, verbose=1,
				validation_data=val_ds)

model.summary()

results = model.evaluate(test_ds)
print('AUC, Accuracy, Precision, Recall, TN, TP, FN, FP:')
print(results)

# save output 
y_pred = model.predict(test_ds)
y_test = tf.concat([y for x,y in test_ds], axis=0)

save_path = 'tf_data_5epoch'
np.savetxt(os.path.join(work_dir, '../Models/LSTM', \
				"{}_y_pred.csv".format(save_path)), y_pred, delimiter=",")
np.savetxt(os.path.join(work_dir, '../Models/LSTM', \
				"{}_y_test.csv".format(save_path)), y_test.numpy(), delimiter=",")
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(work_dir, '../Models/LSTM', \
				"{}_history.csv".format(save_path)), index=False)

os.mkdir(os.path.join(work_dir, '../Models/LSTM', save_path))
model.save(os.path.join(work_dir, '../Models/LSTM', save_path))


print("Batch job ended: ", datetime.now() - startTime)


"""
def evaluation_plotting(history, model, save_path, x_test):

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

def plot_roc(history, y_pred, save_path):

	fpr, tpr, thresholds = roc_curve(y_test.numpy(), y_pred)
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


"""