import io
import csv
import pandas as pd


def inp_file_2_csv(inp_file, index_column, header_row):
    if index_column:
            index_col = 0
    else:
        index_col = None
    
    if header_row:
        header = 0
    else:
        header = None


    bla = io.StringIO(inp_file.getvalue().decode("utf-8"))

    dialect = csv.Sniffer().sniff(bla.read())
    
    df = pd.read_csv(inp_file, dialect=dialect, index_col=index_col, header=header)

    return df