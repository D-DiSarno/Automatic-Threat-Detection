import csv
import json
import logging
import requests
import csv
import tempfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import shlex
from keras_preprocessing.sequence import pad_sequences

import pickle

from datetime import datetime
import joblib

import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.DEBUG)


# File di input
file_name = "/Users/davidedisarno/Desktop/Tesi/TesiMagistraleHoneyNet/MalwareAnalysis/cowrie/var/log/cowrie/cowrie.json"
"""def prepare(input_file):

    with open(input_file, "r+") as infile:
        # Create a csv reader object
        reader = csv.reader(infile)
        # Read the first row as the original headers
        headers = next(reader)
        # Find the index of the input header
        input_index1 = headers.index("input")
        # Change the input header to Commands
        headers[input_index1] = "Commands"
        # Find the index of the input header
        input_index2 = headers.index("session")
        # Change the input header to Commands
        headers[input_index2] = "Session"
        # Create a new list to store the reordered headers
        new_headers = []
        # Append the headers in the desired order
        new_headers.append(headers[input_index2])  # Session
        new_headers.append(headers[1])  # eventid
        new_headers.append(headers[input_index1])  # Commands
        new_headers.append(headers[2])  # src_ip
        new_headers.append(headers[4])  # hour
        new_headers.append(headers[6])  # day
        new_headers.append(headers[7])  # month

        # Create a temporary file object
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        # Create a csv writer object for the temporary file
        writer = csv.writer(temp_file)
        # Write the new headers as the first row
        writer.writerow(new_headers)
        # Loop through the remaining rows in the original file
        for row in reader:
            # Create a new list to store the reordered row
            new_row = []
            # Append the values in the desired order
            new_row.append(row[input_index2])  # Session
            new_row.append(row[1])  # eventid
            new_row.append(row[input_index1])  # Commands
            new_row.append(row[2])  # src_ip
            new_row.append(row[4])  # hour
            new_row.append(row[6])  # day
            new_row.append(row[7])  # month
            # Write the new row to the temporary file
            writer.writerow(new_row)

    # Close both files
    infile.close()
    temp_file.close()
    # Overwrite the original file with the temporary file contents
    with open(input_file, "w") as outfile:
        with open(temp_file.name, "r") as infile:
            outfile.write(infile.read())

"""


def prepare(input_file):
    temp_fd, temp_path = tempfile.mkstemp()
    with open(input_file, "r") as infile, open(temp_path, "w", newline='') as temp_file:
        reader = csv.reader(infile)
        headers = next(reader)
        input_index1 = headers.index("input")
        headers[input_index1] = "Commands"
        input_index2 = headers.index("session")
        headers[input_index2] = "Session"
        new_headers = [headers[input_index2], headers[1], headers[input_index1], headers[2], headers[4], headers[6],
                       headers[7]]
        writer = csv.writer(temp_file)
        writer.writerow(new_headers)
        for row in reader:
            new_row = [row[input_index2], row[1], row[input_index1], row[2], row[4], row[6], row[7]]
            writer.writerow(new_row)

    with open(temp_path, "r") as temp_file, open(input_file, "w", newline='') as outfile:
        outfile.write(temp_file.read())



def parse_timestamp(timestamp):
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    return dt.hour, dt.date(), dt.day, dt.month

def filter_time(input_file):
    # Define the allowed event IDs
    allowed_eventids = ['cowrie.command.input', 'cowrie.login.failed', 'cowrie.login.success',
                        'cowrie.session.file_download']

    # Read the input file into a DataFrame
    df = pd.read_csv(input_file)

    # Filter the DataFrame based on allowed event IDs and non-empty rows
    df = df[df['eventid'].isin(allowed_eventids) & df.apply(lambda x: any(x.str.strip()), axis=1)]

    # Write the filtered DataFrame back to the input file
    df.to_csv(input_file, index=False)
    # Apply the function to the timestamp column and create new columns
    df[["hour", "date", "day", "month"]] = df["timestamp"].apply(lambda x: pd.Series(parse_timestamp(x)))

    # Delete the original timestamp column
    df = df.drop("timestamp", axis=1)

    # Save the modified dataset
    df.to_csv(input_file, index=False)


def destfile_login(input_file):
    # Read the entire input file into memory
    input_data = []
    with open(input_file, 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        for row in reader:
            input_data.append(row)

    # Modify the input data in memory
    fieldnames = ['session', 'eventid', 'src_ip', 'timestamp', 'input']
    modified_data = []
    for row in input_data:
        if row['destfile']:
            row['input'] = row['destfile']
        if row['username'] or row['password']:
            row['input'] = row['username'] + '/' + row['password']
        del row['destfile']
        del row['username']
        del row['password']
        modified_data.append(row)

    # Write the modified data back to the input file
    with open(input_file, 'w', newline='') as csv_in:
        writer = csv.DictWriter(csv_in, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(modified_data)


def convert_json_lines_to_csv(file_name, output_file_name):
    # specify the columns to keep in the output CSV file
    columns_to_keep = ['session', 'eventid', 'src_ip', 'destfile', 'username', 'password', 'timestamp', 'input']

    # read in the JSON data from the input file
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            data.append(json_line)

    # write the filtered data to the output CSV file
    with open(output_file_name, 'w', newline='') as f:
        writer = csv.writer(f)

        # write the header row with the selected column names
        writer.writerow(columns_to_keep)

        # iterate over each item in the JSON data and write a row to the CSV file
        for item in data:
            row = [item.get(col, '') for col in columns_to_keep]
            writer.writerow(row)

    # print a success message
    print(f"Converted {file_name} from JSON to CSV and kept only {columns_to_keep}")


def commands_extract(input_file):
    input_data = []
    with open(input_file, "r") as infile:
        reader = csv.reader(infile)
        for row in reader:
            input_data.append(row)

    # Modify the input data in memory
    modified_data = []
    header_row = input_data.pop(0)
    modified_data.append(header_row)
    for row in input_data:
        if row:
            if not row[-1]:
                row[-1] = "None"
            input_str = row[7]
            if input_str:
                commands = parse_commands(input_str)
                for command in commands:
                    output_row = [
                        row[0],  # session
                        row[1],  # eventid
                        row[2],  # src_ip
                        row[3],  # destfile
                        row[4],  # username
                        row[5],  # password
                        row[6],  # timestamp
                        command[0]  # command
                    ]
                    modified_data.append(output_row)
            else:
                row[-1] = ''
                modified_data.append(row)

    # Write the modified data back to the input file
    with open(input_file, "w", newline='') as infile:
        writer = csv.writer(infile)
        writer.writerows(modified_data)

def parse_commands(input_str):
    commands = []
    lexer = shlex.shlex(input_str)
    lexer.wordchars += "|>"
    while True:
        token = lexer.get_token()
        if not token:
            break
        if token == "|":
            continue
        else:
            command = []
            command.append(token)
            while True:
                next_token = lexer.get_token()
                lexer.push_token(next_token)
                if not next_token or next_token == "|":
                    break
                else:
                    parameter = lexer.get_token()
                    command.append(parameter)
            commands.append(command)
    return commands

def send_to_splunk(results):
    logging.debug(f"Inizio invio risultati a Splunk: {results}")
    print(f"Inizio invio risultati a Splunk: {results}")
    splunk_url = "http://localhost:8088/services/collector/event"
    splunk_token = "ff32afb4-f85c-4359-a332-76fa43d07051"
    headers = {'Authorization': f'Splunk {splunk_token}'}
    data = {
        "event": {
            "results": results
        },
        "sourcetype": "json"
    }
    response = requests.post(splunk_url, headers=headers, json=data)
    if response.status_code == 200:
        logging.debug("Results sent to Splunk successfully.")
        print("Results sent to Splunk successfully.")
    else:
        logging.error(f"Failed to send results to Splunk: {response.text}")
        print(f"Failed to send results to Splunk: {response.text}")


if __name__ == "__main__":
    output_file = "final_output.csv"
    # Step 1: Convert
    convert_json_lines_to_csv(file_name,output_file)
    print("Step 1: Conversione completata.")

    # Step 2: Extract
    commands_extract(output_file)
    print("Step 2: Estrazione completata.")

    # Step 3: Destfile
    destfile_login(output_file)
    print("Step 3: Destfile completata.")

    # Step 4: Filter
    filter_time(output_file)
    print("Step 4: Filtraggio completato.")

    # Step 5: Prepare
    prepare(output_file)
    print("Step 5: Preparazione completata.")

    # Output finale

    with open(output_file, 'r') as f:
        results = list(csv.DictReader(f))

    print(f"Elaborazione completata. File di output salvato come {output_file}")


    # Predict and handle zero-day attacks
    with open("model/label_encoder_1.pkl", "rb") as f:
        le = joblib.load(f)  # Load the LabelEncoder using joblib

    tokenizer = joblib.load("model/tokenizer_1.pkl")
    enc = joblib.load("model/encoder_1.pkl")
    with open("model/max_sequence_length_1.pkl", 'rb') as f:
        max_sequence_length = pickle.load(f)

        # Load individual models
        # Load individual models using TensorFlow SavedModel format
        bdlstm_model = tf.keras.models.load_model("model/bdlstm_model_1")
        gru_model = tf.keras.models.load_model("model/gru_model_1")
        cnn_model = tf.keras.models.load_model("model/cnn_model_1")


        # Recreate the voting ensemble
        class VotingEnsemble:
            def __init__(self, models):
                self.models = models

            def predict_proba(self, X):
                predictions = np.array([model.predict(X) for model in self.models])
                avg_predictions = np.mean(predictions, axis=0)
                return avg_predictions


        ensemble_model = VotingEnsemble([bdlstm_model, gru_model, cnn_model])

    # Verify the LabelEncoder
    print("LabelEncoder classes:", le.classes_)
    print("Is instance of LabelEncoder:", isinstance(le, LabelEncoder))

    # Load the new data
    new_data = pd.read_csv("final_output.csv")
    original_commands = new_data['Commands'].copy()

    # Preprocess the data
    new_data['Commands'] = new_data['Commands'].str.replace('[^\w\s]', '', regex=True)

    # Tokenize the Commands column
    sequences = tokenizer.texts_to_sequences(new_data['Commands'])
    X_new = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    # Encode categorical features
    enc_src_ip = enc.transform(new_data[['src_ip']]).toarray()

    # Concatenate numeric and categorical features
    X_new = np.column_stack((X_new, enc_src_ip, new_data[['hour', 'day', 'month']].values))

    # Make predictions for the new data
    y_pred = ensemble_model.predict_proba(X_new)


    # Handle zero-day attacks
    def handle_zero_day(probs):
        sum_known_probs = np.sum(probs, axis=1)
        zero_day_probs = 1 - sum_known_probs
        zero_day_probs = np.clip(zero_day_probs, 0, 1)
        return zero_day_probs


    zero_day_probs = handle_zero_day(y_pred)
    predicted_indices = np.argmax(y_pred, axis=1)

    # Ensure that le is a LabelEncoder and perform inverse transform
    if isinstance(le, LabelEncoder):
        predicted_tactics = le.inverse_transform(predicted_indices)
    else:
        raise TypeError("The loaded object is not a LabelEncoder")

    # Add the predicted tactics to the DataFrame
    new_data['predicted_tactic'] = predicted_tactics
    new_data['Commands'] = original_commands

    # Save the DataFrame with the predicted tactics
    new_data.to_csv("cowrie_completo_preprocessing.csv", index=False)

    print("Prediction and zero-day detection completed.")

    # Send results to Splunk
    send_to_splunk(new_data.to_dict(orient='records'))
