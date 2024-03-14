csv_file_path = 'dataset/steam_spy_data.csv'

# Read the CSV file as text
with open(csv_file_path, 'r') as file:
    content = file.read()

# Remove unwanted characters (e.g., semicolons)
content_without_semicolons = content.replace(';', '')

# Write the modified content back to the file
with open(csv_file_path, 'w') as file:
    file.write(content_without_semicolons)