import os
current_path = os.path.abspath(os.path.dirname(__file__))
filelist_path = current_path + "/filelist.txt"

filelist = open(filelist_path, "w")
for filename in os.listdir(current_path):
    if filename.split(".")[1] == "mp4":
        if filename.find("[") != -1:
            print(filename.find("["))
            newname = filename[:filename.find("[")] + ".mp4"
            os.rename(filename, newname)
            filelist.write("file " + newname + "\n")
        else:
            filelist.write("file " + filename + "\n")

filelist.close()
