# data_mining

DM_Cleaning file is for cleaning the 9 train files that I've got fter runing Magda's code.
The code does the following:
1- Drop some columns such as accuracy, views, licenseID and DateUploaded.
2- Removes panctuations, '\n', '\r', any text starts with user id, ip, http links or any non ascii characters from the photo tags
3- Convert all letters to lower case
4- finally combines the 9 files into one file clled train_cleaned.csv
