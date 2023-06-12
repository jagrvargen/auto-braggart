import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Use the JSON key you downloaded to authenticate and create an API client
scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('Your_Json_File_Name.json', scope)
client = gspread.authorize(creds)

# Replace 'Your_Sheet_Name' with the actual name of the Google Sheet you want to access
sheet = client.open('Your_Sheet_Name').sheet1

# Fetch all the records
data = sheet.get_all_records()

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)
