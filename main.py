import reconstruction as re
import DataframeCreator as dfc


def main ():

    test_files_directory = 'test_pics/Downloaded Persian Miniatures - Cropped and Resized/'
    output_files_directory = 'output/'
    excel_directory = 'data.xlsx'


    df = re.convertExcel2Dataframe (excel_file_path=excel_directory)

    print ('Task ended successfully.')


if __name__ == "__main__":
    main ()
