import os


base_dir = r'C:\Users\Rohit Negi\Desktop\Projects\Major Project\Acute Lymphoblastic Leukemia dataset\Split_Data'

# Function to count files in each folder and subfolder
def count_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        # Print the folder path and the number of files in it
        print(f"{root}: {len(files)} files")

# Call the function
count_files_in_directory(base_dir)