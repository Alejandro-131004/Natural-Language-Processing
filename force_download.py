import os
import ssl
import subprocess

# Force disable SSL verification globally in this script
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

def download_dataset():
    dataset_id = "akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000"
    print(f"Attempting to download {dataset_id} ignoring SSL...")
    
    # We call the command-line tool via subprocess
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_id], check=True)
        print("Download request sent successfully.")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_dataset()