import reconstruction as re
import os
import pandas as pd
import DataframeCreator as dfc

def fetchFilenames (file_directory):
    return os.listdir (file_directory)

def main ():

    test_files_directory = 'test_pics/Downloaded Persian Miniatures - Cropped and Resized'
    output_files_directory = 'output/'


    if os.path.isfile ('Acoefs.xlsx'):
        df = pd.read_excel ('Acoefs.xlsx')
    else:
        df = pd.DataFrame ({'file name':[], 'acoefs':[], 'aCoefs_optimized':[]})
        dfc.appendRow (df, 'blank', None, None)
        dfc.exportDataFrameToExcel (df)
        

    for filename in os.listdir (test_files_directory):
        file_path = os.path.join (test_files_directory, filename)

        print (filename)

        if os.path.exists (output_files_directory + filename):
            print ('Output file already exists. Skipped.')
            continue

        re.create_report (file_path, filename, df)
        dfc.exportDataFrameToExcel (df)
        


    print ('Task ended successfully.')


if __name__ == "__main__":
    main ()
