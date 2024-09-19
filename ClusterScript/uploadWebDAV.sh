#!/bin/bash

# WebDAV credentials
WEBDAV_URL="https://surfdrive.surf.nl/files/remote.php/webdav/"  # Your WebDAV URL
USERNAME="shankaras1@leidenuniv.nl"  # Replace with your WebDAV username
PASSWORD="CWDAK-EIELG-UZVNQ-ATKVV"  # Replace with your WebDAV password

# Directory to upload
LOCAL_DIRECTORY="./Dump"  # Replace with the path to your local directory

# Extract the base name of the local directory
BASE_DIR_NAME=$(basename "$LOCAL_DIRECTORY")

# Function to recursively upload files and directories
upload_files() {
    local local_dir="$1"
    local remote_dir="$2"

    # Create the remote directory in SurfDrive
    curl -u "$USERNAME:$PASSWORD" -X MKCOL "$WEBDAV_URL$remote_dir"

    # Loop through all files and directories in the current local directory
    for item in "$local_dir"/*; do
        # Get the base name of the file/directory
        local base_name=$(basename "$item")

        if [ -d "$item" ]; then
            # If it's a directory, recursively upload it
            echo "Creating and uploading directory: $item"
            upload_files "$item" "$remote_dir/$base_name"
        elif [ -f "$item" ]; then
            # If it's a file, upload it
            echo "Uploading file: $item"
            curl -u "$USERNAME:$PASSWORD" -T "$item" "$WEBDAV_URL$remote_dir/$base_name"
        fi
    done
}

# Create the top-level directory in SurfDrive
echo "Creating remote directory: $BASE_DIR_NAME"
curl -u "$USERNAME:$PASSWORD" -X MKCOL "$WEBDAV_URL$BASE_DIR_NAME"

# Start uploading the entire directory
upload_files "$LOCAL_DIRECTORY" "$BASE_DIR_NAME"

echo "Upload complete!"

