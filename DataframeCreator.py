import pandas as pd

def initializeEmptyDataframe (column_names):
    '''
        Initializes an empty pandas DataFrame with appropriate column names.

        Returns:
            pd.DataFrame: Empty DataFrame.
    '''

    df = pd.DataFrame (columns= column_names)

    return df


def exportDataFrameToExcel (df, output_file="Acoefs.xlsx", sheet_name="Sheet1"):
    '''
        Exports a pandas DataFrame to an Excel file.

        Args:
            df (pd.DataFrame) : DataFrame to export.
            output_file (str, optional): Output Excel file path. Defaults to "Acoefs.xlsx".
    '''

    with pd.ExcelWriter (output_file) as writer:
        df.to_excel (writer, sheet_name=sheet_name, index=False)


def appendRow (df, file_name, acoefs, acoefs_optimized):

    df.loc [len (df.index)] = [file_name, acoefs, acoefs_optimized]