import os

import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from keras.layers import Bidirectional, Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, GRU, Flatten
from tensorflow.python.keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
# Load the dataset
data = pd.read_csv("../data/cowrie_predict_completo.csv")



# Preprocess the data
data['Commands'] = data['Commands'].str.replace('[^\w\s]', '')

# Tokenize the Commands column
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Commands'])
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(data['Commands'])
max_sequence_length = max([len(seq) for seq in sequences])
X = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Encode categorical features
enc = OneHotEncoder(handle_unknown='ignore')
enc_src_ip = enc.fit_transform(data[['src_ip']]).toarray()

# Encode the 'Tactic' column as integer labels
le = LabelEncoder()
data['Tactic'] = le.fit_transform(data['Tactic'])

# Concatenate numeric and categorical features
X = np.column_stack((X, enc_src_ip, data[['hour', 'day', 'month']]))

# Split the dataset into training and testing sets
y = to_categorical(data['Tactic'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model definitions
def create_bdlstm_model():
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_sequence_length + enc_src_ip.shape[1] + 3))
    model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, kernel_regularizer=l2(0.1))))
    model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l2(0.1))))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(len(np.unique(data['Tactic'])), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model

def create_cnn_model_v2():
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_sequence_length + enc_src_ip.shape[1] + 3))
    model.add(Conv1D(16, kernel_size=4, padding='same', activation='relu', kernel_regularizer=l2(0.1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.8))
    model.add(Conv1D(32, kernel_size=4, padding='same', activation='relu', kernel_regularizer=l2(0.1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.1)))
    model.add(Dropout(0.8))
    model.add(Dense(len(np.unique(data['Tactic'])), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model

def create_gru_model_v2():
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_sequence_length + enc_src_ip.shape[1] + 3))
    model.add(GRU(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, kernel_regularizer=l2(0.1)))
    model.add(GRU(128, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=l2(0.1)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(len(np.unique(data['Tactic'])), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model

# Oversample the minority classes using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, np.argmax(y_train, axis=1))
y_resampled = to_categorical(y_resampled)

# Instantiate the models
bdlstm_model = create_bdlstm_model()
cnn_model = create_cnn_model_v2()
gru_model = create_gru_model_v2()

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Cross-validation training
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for train, val in kfold.split(X_resampled, y_resampled):
    bdlstm_model.fit(X_resampled[train], y_resampled[train], validation_data=(X_resampled[val], y_resampled[val]), epochs=100, batch_size=64, callbacks=[reduce_lr, early_stopping])
    cnn_model.fit(X_resampled[train], y_resampled[train], validation_data=(X_resampled[val], y_resampled[val]), epochs=100, batch_size=64, callbacks=[reduce_lr, early_stopping])
    gru_model.fit(X_resampled[train], y_resampled[train], validation_data=(X_resampled[val], y_resampled[val]), epochs=100, batch_size=64, callbacks=[reduce_lr, early_stopping])


# Evaluate each model separately and print the classification report
def evaluate_and_print_report(model, X_train, y_train, X_test, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_train_pred_classes = np.argmax(y_train_pred, axis=1)
    y_train_classes = np.argmax(y_train, axis=1)

    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    print(f"Classification report for {model_name} (Training Data):")
    print(classification_report(y_train_classes, y_train_pred_classes, zero_division=1))
    train_accuracy = accuracy_score(y_train_classes, y_train_pred_classes)
    train_precision = precision_score(y_train_classes, y_train_pred_classes, average='weighted', zero_division=1)
    train_recall = recall_score(y_train_classes, y_train_pred_classes, average='weighted')
    print(f"Training Accuracy for {model_name}: {train_accuracy}")
    print(f"Training Precision for {model_name}: {train_precision}")
    print(f"Training Recall for {model_name}: {train_recall}")

    print(f"Classification report for {model_name} (Testing Data):")
    print(classification_report(y_test_classes, y_test_pred_classes, zero_division=1))
    test_accuracy = accuracy_score(y_test_classes, y_test_pred_classes)
    test_precision = precision_score(y_test_classes, y_test_pred_classes, average='weighted', zero_division=1)
    test_recall = recall_score(y_test_classes, y_test_pred_classes, average='weighted')
    print(f"Testing Accuracy for {model_name}: {test_accuracy}")
    print(f"Testing Precision for {model_name}: {test_precision}")
    print(f"Testing Recall for {model_name}: {test_recall}")

    return (train_accuracy, train_precision, train_recall), (test_accuracy, test_precision, test_recall)


bdlstm_train_metrics, bdlstm_test_metrics = evaluate_and_print_report(bdlstm_model, X_train, y_train, X_test, y_test, "BDLSTM")
cnn_train_metrics, cnn_test_metrics = evaluate_and_print_report(cnn_model, X_train, y_train, X_test, y_test, "CNN")
gru_train_metrics, gru_test_metrics = evaluate_and_print_report(gru_model, X_resampled, y_resampled, X_test, y_test, "GRU")

# Combine the models into a voting classifier

class VotingEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        avg_predictions = np.mean(predictions, axis=0)
        return np.argmax(avg_predictions, axis=1)

# Create the voting ensemble
ensemble = VotingEnsemble([bdlstm_model, cnn_model, gru_model])

# Make predictions for the test set
y_pred_classes = ensemble.predict(X_test)
y_train_pred_classes = ensemble.predict(X_train)

# Convert one-hot encoded y_test to class labels
y_test_classes = np.argmax(y_test, axis=1)
y_train_classes = np.argmax(y_train, axis=1)

# Print classification report for the ensemble
print("Classification report for Voting Ensemble (Training Data):")
print(classification_report(y_train_classes, y_train_pred_classes, zero_division=1))
ensemble_train_accuracy = accuracy_score(y_train_classes, y_train_pred_classes)
ensemble_train_precision = precision_score(y_train_classes, y_train_pred_classes, average='weighted', zero_division=1)
ensemble_train_recall = recall_score(y_train_classes, y_train_pred_classes, average='weighted')
print(f"Training Accuracy for Voting Ensemble: {ensemble_train_accuracy}")
print(f"Training Precision for Voting Ensemble: {ensemble_train_precision}")
print(f"Training Recall for Voting Ensemble: {ensemble_train_recall}")

print("Classification report for Voting Ensemble (Testing Data):")
print(classification_report(y_test_classes, y_pred_classes, zero_division=1))
ensemble_test_accuracy = accuracy_score(y_test_classes, y_pred_classes)
ensemble_test_precision = precision_score(y_test_classes, y_pred_classes, average='weighted', zero_division=1)
ensemble_test_recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
print(f"Testing Accuracy for Voting Ensemble: {ensemble_test_accuracy}")
print(f"Testing Precision for Voting Ensemble: {ensemble_test_precision}")
print(f"Testing Recall for Voting Ensemble: {ensemble_test_recall}")

# Print individual model metrics
print("--------")
print("BDLSTM Metrics (Training): ", bdlstm_train_metrics)
print("BDLSTM Metrics (Testing): ", bdlstm_test_metrics)
print("--------")
print("CNN Metrics (Training): ", cnn_train_metrics)
print("CNN Metrics (Testing): ", cnn_test_metrics)
print("--------")
print("GRU Metrics (Training): ", gru_train_metrics)
print("GRU Metrics (Testing): ", gru_test_metrics)
print("--------")
print("Voting Ensemble Metrics (Training): ", (ensemble_train_accuracy, ensemble_train_precision, ensemble_train_recall))
print("Voting Ensemble Metrics (Testing): ", (ensemble_test_accuracy, ensemble_test_precision, ensemble_test_recall))
print("--------")

models = {
    'BDLSTM': bdlstm_model,
    'Improved GRU': gru_model,
    'Improved CNN': cnn_model
}

for name in models:
    reduced_models = [model for model_name, model in models.items() if model_name != name]
    reduced_ensemble = VotingEnsemble(reduced_models)
    y_pred_reduced = reduced_ensemble.predict(X_test)
    reduced_score = accuracy_score(y_test_classes, y_pred_reduced)
    print(f"Accuracy without {name}: {reduced_score}")


bdlstm_model.save("bdlstm_model_1")
gru_model.save("gru_model_1")
cnn_model.save("cnn_model_1")

ensemble_config = {
    'models': ['bdlstm_model', 'cnn_model','gru_model']
}

joblib.dump(tokenizer, "tokenizer_1.pkl")
joblib.dump(enc, "encoder_1.pkl")
with open("max_sequence_length_1.pkl", "wb") as f:
    pickle.dump(max_sequence_length, f)
with open("label_encoder_1.pkl", "wb") as f:
    pickle.dump(le, f)

print("Ensemble model and related objects have been saved.")
