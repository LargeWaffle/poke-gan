import os

folder = "all"
for count, filename in enumerate(os.listdir(folder)):
    dst = f"{str(count)}.png"
    src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst = f"res/{dst}"
    os.rename(src, dst)
