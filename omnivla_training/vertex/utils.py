import os
import re
from datetime import datetime
from typing import Tuple

import google_crc32c
from google.cloud import secretmanager, storage


def upload_file(file_path: str, job_dir: str, project_id: str, file_name: str) -> str:
    """
    Uploads a file to the job directory in GCS.
    
    Args:
        file_path: LOCAL Path to a file.
        job_dir: GCS path to the job directory.
        project_id: GCP project ID.
        file_name: Name of the file to be stored in GCS with extension.

    Returns:  the remote path where the file is stored.
    """
    regex = re.compile(r"gs://([a-z0-9-]+)/(.*)")
    match = regex.match(job_dir)
    bucket_name = match.group(1)
    remote_path = match.group(2)

    client = storage.Client(project=project_id)
    bucket = client.get_bucket(bucket_name)

    blob_path = os.path.join(remote_path, file_name)
    blob = bucket.blob(blob_path)
    print(f"Uploading {file_path} to {bucket_name}/{blob_path}...")
    blob.upload_from_filename(file_path)

    return os.path.join(job_dir, file_name)


def upload_config(config_path: str, job_dir: str, project_id: str) -> str:
    """
    Uploads the training configuration to the job directory.
    
    Args:
        config_path: LOCAL Path to the config file.
        job_dir: GCS path to the job directory.
        project_id: GCP project ID.

    Returns:  the remote path where the config is stored.
    """
    return upload_file(config_path, job_dir, project_id, "train_config.yaml")


def access_secret_version(
    secret_id: str, version_id: str, project_id: str
) -> Tuple[bool, bytes]:
    """
    Access the payload for the given secret version if one exists. The version
    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    
    Args:
        secret_id: The secret ID in GCP Secret Manager.
        version_id: The version ID (e.g., "1" or "latest").
        project_id: GCP project ID.
    
    Returns:
        Tuple of (success: bool, payload: bytes)
    """

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    response = client.access_secret_version(request={"name": name})

    # Verify payload checksum.
    crc32c = google_crc32c.Checksum()
    crc32c.update(response.payload.data)
    success = True
    if response.payload.data_crc32c != int(crc32c.hexdigest(), 16):
        print("Data corruption detected.")
        success = False

    return success, response.payload.data


def get_gituser() -> str:
    """
    Returns the git user name.
    """
    try:
        return os.popen("git config --get user.name").read().strip()
    except Exception:
        return "unknown-user"


def get_timestamp() -> str:
    """
    Returns the current timestamp.
    """
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


