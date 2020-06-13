import dropbox

from tqdm.cli import tqdm
from pathlib import Path


data_folder = Path("data")
archive_folder = Path("archive")
sparse_segm_folder = data_folder / "sparse-segm"

token_filename = archive_folder / "dropbox_token.txt"
with open(token_filename) as token_file:
    TOKEN = token_file.read()
dbx = dropbox.Dropbox(TOKEN)


folder_name = "/Openedsdata2020/openEDS2020-SparseSegmentation/participant/"            

for entry in tqdm(dbx.files_list_folder(folder_name).entries):
    output_filename = str(sparse_segm_folder / entry.path_display.replace(folder_name, ""))
    output_filename = str(output_filename)
    if isinstance(entry, dropbox.files.FolderMetadata):
        dbx.files_download_zip_to_file(output_filename + ".zip", entry.path_lower)
    else:
        dbx.files_download_to_file(output_filename, entry.path_lower)

