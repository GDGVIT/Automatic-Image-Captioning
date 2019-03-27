import zipfile
location = "/content/gdrive/My Drive/flickr_dataset/glove.6B.zip"
zip_ref = zipfile.ZipFile(location, 'r')
zip_ref.extractall("/content/gdrive/My Drive/flickr_dataset/glove.6B/")
zip_ref.close()

import zipfile
location = "/content/gdrive/My Drive/flickr_dataset/Flickr8k_Dataset.zip"
zip_ref = zipfile.ZipFile(location, 'r')
zip_ref.extractall("/content/gdrive/My Drive/flickr_dataset/Flickr8k_Dataset/")
zip_ref.close()

!pip install -U -q PyDrive

zip_id = ''

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import zipfile, os


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

if not os.path.exists('MODEL'):
    os.makedirs('MODEL')

# DOWNLOAD ZIP
print ("Downloading zip file")
#downloaded = drive.CreateFile({'id': uploaded.get('id')})
myzip = drive.CreateFile({'id': uploaded.get(zip_id)})
myzip.GetContentFile('Flickr8k_text.zip')

# UNZIP ZIP
print ("Uncompressing zip file")
zip_ref = zipfile.ZipFile('Flickr8k_text.zip', 'r')
zip_ref.extractall('MODEL/')
zip_ref.close()

from google.colab import drive
drive.mount('/content/gdrive')

!wget -cq ""

!cd ""

!unzip "/content/gdrive/'My Drive'/flickr_dataset/Flickr8k_text.zip" "/content/gdrive/'My Drive'/flickr_dataset/exp/"

!ls

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

import zipfile
zip_ref = zipfile.ZipFile("Flickr8k_text.zip", 'r')
zip_ref.extractall("Flickr8k_text")
zip_ref.close()