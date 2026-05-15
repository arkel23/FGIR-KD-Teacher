import os
import glob
import argparse
from huggingface_hub import login, create_repo, HfApi
from pathlib import Path

def main(folder, extension, mytoken, repos):

    # take the target folder
    path = Path(os.getcwd() + "/" + folder)
    print(f"path is {path}")

    # list all the files with the specified extension
    try:
        file_paths = list(path.glob('*' + extension))

        if not file_paths:
            raise FileNotFoundError(f"no file with {extension} found. Terminate script!")
        
        file_list = [file_path.name for file_path in file_paths]
        
        print(f"we found {len(file_list)} files")

    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    # Log into Hugging Face Hub
    login(token=mytoken)

    # upload file
    api = HfApi()
    
    print(f"target repos: {repos}")

    # upload file one by one
    for i, myfile in enumerate(file_list):
        try:
            api.upload_file(
                path_or_fileobj = path / myfile,
                path_in_repo = myfile,
                repo_id = repos,
                repo_type = "model",
            )
            print(f"File {i+1}: \"{myfile}\" uploaded successfully.")
        except Exception as e:
            print(f"Error uploading file: {e}")
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a file to Hugging Face Hub")

    parser.add_argument('-f', '--folder', type=str, required=True, help='folder you want to upload')
    parser.add_argument('-t', '--mytoken', type=str, required=True, help='Hugging Face API token')
    parser.add_argument('-e', '--ext', type=str, required=True, help='file extension you want to upload. example = .py')
    parser.add_argument('-r', '--repos', type=str, required=True, help='Repository name')

    args = parser.parse_args()

    main(args.folder, args.ext, args.mytoken, args.repos)

