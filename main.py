import reconstruction as re
import pandas as pd
import DataframeCreator as dfc


def main ():

    test_files_directory = 'test_pics/Downloaded Persian Miniatures - Cropped and Resized/'
    output_files_directory = 'output/'
    excel_directory = 'Acoefs.xlsx'


    df = pd.read_excel (excel_directory)
    df = re.addGaussianParsToDataFrame (re.convertExcel2Dataframe (excel_directory), test_files_directory)
    dfc.exportDataFrameToExcel (df, 'data.xlsx')

    print ('Task ended successfully.')


if __name__ == "__main__":
    main ()
