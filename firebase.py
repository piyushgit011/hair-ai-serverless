import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate("/workspace/hair_ai/Hair AI Firebase Admin.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'hai-style-with-ai.appspot.com'
})

def convert_to_url(local_file, cloud_file):
    bucket = storage.bucket()
    blob = bucket.blob(cloud_file)
   # blob.upload_from_filename(local_file)
    # Make the blob publicly viewable
    local_file.seek(0)  # Ensure the pointer is at the start of BytesIO object
    blob.upload_from_file(local_file, content_type='image/png')
    blob.make_public()
    # Return the public URL
    return blob.public_url
