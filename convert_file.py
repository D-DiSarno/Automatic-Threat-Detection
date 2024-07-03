import json
import csv


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



# Define the input and output file names
file_name = "/Users/davidedisarno/Desktop/Tesi/TesiMagistraleHoneyNet/MalwareAnalysis/cowrie/var/log/cowrie/cowrie.json"
#file_name="cowrie_secondoDataset.csv"
output_file_name = "cowrie_convertito.csv"

# Call the function to convert JSON lines to CSV
convert_json_lines_to_csv(file_name, output_file_name)
