from io import BytesIO
import csv
import pandas as pd


def inp_file_2_csv(inp_file, index_column, header_row, delimiter=None):
    try:
        # Determine index_col and header based on input parameters
        index_col = 0 if index_column else None
        header = 0 if header_row else None

        # Read the uploaded file into a BytesIO object
        file_content = inp_file.read()

        # Check if the file is empty
        if not file_content:
            raise ValueError("The uploaded file is empty.")

        # Read a portion of the file content to sniff the dialect
        sample = file_content[:1024].decode('utf-8')

        # Sniff the dialect
        dialect = csv.Sniffer().sniff(sample)

        # Create a BytesIO object from the file content
        file_like_object = BytesIO(file_content)
        if delimiter:
            df = pd.read_csv(file_like_object, delimiter=delimiter, index_col=index_col, header=header)
        else: 
        # Load the CSV into a DataFrame
            df = pd.read_csv(file_like_object, dialect=dialect, index_col=index_col, header=header)

        return df

    except csv.Error as e:
        raise ValueError(f"Error parsing CSV file: {e}")
    except pd.errors.EmptyDataError:
        raise ValueError("No columns to parse from file. The file might be empty or the header row is not correctly identified.")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")