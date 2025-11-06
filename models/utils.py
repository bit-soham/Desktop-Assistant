# Function to open a file and return its contents as a string
def open_file(filepath):
    print(f"DEBUG: Opening file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()