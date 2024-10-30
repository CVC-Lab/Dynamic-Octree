import json

def filter_and_save_json_range(input_file_path, output_file_path, start_key, end_key):
    """Filter keys from a JSON file and save the specified range to another JSON file.

    Args:
        input_file_path (str): Path to the input JSON file.
        output_file_path (str): Path to the output JSON file.
        start_key (str): The starting key in the range.
        end_key (str): The ending key in the range.
    """
    # Load data from the input JSON file
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # Filter the data for the specified range
    filtered_data = {key: data[key] for key in data if int(start_key) <= int(key) <= int(end_key)}

    # Save the filtered data to the output JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)

# Example usage
filter_and_save_json_range('0.json', '4000.json', "4750", "4999")
