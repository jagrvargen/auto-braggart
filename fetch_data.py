import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Use the JSON key you downloaded to authenticate and create an API client
scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('./data/auto-braggart-9603339e3c33.json', scope)
client = gspread.authorize(creds)

def fetch_all_worksheets():
    sheet = client.open('Formatted Brag Doc - Jesse')
    worksheets = sheet.worksheets()

    df = pd.DataFrame()

    # Iterate over all worksheets and fetch records
    for worksheet in worksheets:
        data = worksheet.get_all_records()
        df_temp = pd.DataFrame(data)

        # Append the data to the main DataFrame
        df = pd.concat([df, df_temp])

    # Save the DataFrame to a CSV file
    df.to_csv('data/brag-doc.csv', index=False)

def fetch_second_worksheet():
    # Replace 'Your_Sheet_Name' with the actual name of the Google Sheet you want to access
    spreadsheet = client.open('Formatted Brag Doc - Jesse')

    # Fetch the second worksheet (0-based index, so index 1 corresponds to the second sheet)
    sheet = spreadsheet.get_worksheet(1)

    # Fetch all the records
    data = sheet.get_all_records()

    # Convert the data to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('data/brag-doc.csv', index=False)

def is_title(text):
    return not text.startswith("You")

def generate_category_str():
    category_str = ""
    sheet = client.open('Formatted Brag Doc - Jesse')
    worksheet = sheet.get_worksheet(1)
    data = worksheet.get_all_records()

    for row in data:
        if not is_title(row['Accomplishments']):
            category_str += row['Accomplishments'] + "\n\n"

    return category_str

def live_update():
    sheet = client.open('Formatted Brag Doc - Jesse')
    worksheet = sheet.get_worksheet(1)
    data = worksheet.get_all_records()

    for index, row in enumerate(data, start=1):
        print(row)
        # Skip title rows based on some condition, for example if the row is empty
        if row['Accomplishments'] == '' or is_title(row['Accomplishments']):  # define the is_title function to determine whether the row is a title
            continue

        # Get the existing answer
        existing_answer = row['Notes']

        # Append your text
        new_answer = existing_answer + ' Your text here.'

        # Write back the new answer to the second column (Google Sheets is 1-indexed)
        worksheet.update_cell(index + 1, 2, new_answer)
