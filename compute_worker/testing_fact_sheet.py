#!/usr/bin/env python3
import requests
import os
import json
import tempfile
import zipfile

# --------------------------
# CONFIGURE THESE
# --------------------------
SUBMISSION_ID = 357821
SECRET = "9219c4f6-8d61-4b91-9473-c0ef33c15b46"
OUTPUT_DIR = os.getcwd()  # change if you want a different folder
DOWNLOAD_ZIP = True       # set to False if you don't want a zip

# --------------------------
# END CONFIGURATION
# --------------------------

HEADERS = {"Authorization": f"Secret {SECRET}"}
BASE_URL = "https://www.codabench.org/api/submissions/"

def fetch_submission_details(submission_id):
    url = f"{BASE_URL}{submission_id}/get_details/"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch submission info: {resp.status_code}\n{resp.text}")
    return resp.json()

def download_file(url, save_path):
    # In some dev environments, you might need url.replace("docker.for.mac.", '')
    resp = requests.get(url)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)

def main():
    data = fetch_submission_details(SUBMISSION_ID)
    print(f"Submission info for ID {SUBMISSION_ID}:")
    print(json.dumps(data, indent=2))

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name

    # Download main submission file
    if "data_file" in data and data["data_file"]:
        download_file(data["data_file"], os.path.join(temp_path, "submission_file.zip"))

    # Download prediction and scoring outputs
    for key in ["prediction_result", "scoring_result"]:
        if key in data and data[key]:
            download_file(data[key], os.path.join(temp_path, f"{key}.zip"))

    # Download logs
    for log in data.get("logs", []):
        filename = f"{log['name']}.txt"
        download_file(log["data_file"], os.path.join(temp_path, filename))

    # Save JSON details
    with open(os.path.join(temp_path, "submission_detail.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if DOWNLOAD_ZIP:
        zip_path = os.path.join(OUTPUT_DIR, f"submission_{SUBMISSION_ID}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(temp_path):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)
        print(f"All files saved to ZIP: {zip_path}")
    else:
        print(f"All files downloaded to temp folder: {temp_path}")

    temp_dir.cleanup()

if __name__ == "__main__":
    main()
