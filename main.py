import reconstruction as re
import os


test_files_directory = 'test_pics/Temp'
output_files_directory = 'output/'


for filename in os.listdir (test_files_directory):
    file_path = os.path.join (test_files_directory, filename)

    print (filename)

    if os.path.exists (output_files_directory + filename):
        print ('Output file already exists. Skipped.')
        continue

    re.create_report (file_path, filename)


print ('Task ended successfully.')
