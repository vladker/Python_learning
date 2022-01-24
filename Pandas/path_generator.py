import os
path_to_file='C:/Users/rogac/Downloads/Dictionaries'
list_files=list(os.listdir(path_to_file))
file_target = open("C:/Users/rogac/Downloads/DICS/dic_list3.dic", mode="a+", encoding="latin-1")
for element in list_files:
    element="'" + path_to_file+'/'+element+"',\n"
    file_target.write(element)
file_target.close()