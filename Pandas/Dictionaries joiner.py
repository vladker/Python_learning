files_locations=['C:/Users/rogac/Downloads/rockyou/rockyou.txt',
                 'C:/Users/rogac/Downloads/DICS/english.dic',
                 'C:/Users/rogac/Downloads/DICS/allwords2.dic',
                 'C:/Users/rogac/Downloads/DICS/beale.dic',
                 'C:/Users/rogac/Downloads/DICS/rus_eng.dic',
                 'C:/Users/rogac/Downloads/DICS/russian.dic',
                 'C:/Users/rogac/Downloads/DICS/unabr.dic',
                 'C:/Users/rogac/Downloads/DICS/BigDict/length08.dic',
                 'C:/Users/rogac/Downloads/Dictionaries/Abbreviations.dic]
file_target = open("C:/Users/rogac/Downloads/DICS/totality.dic", mode="a+", encoding="latin-1")
for file in files_locations:
    file_source=open(file, mode="r", encoding="latin-1")
    for line in file_source:
        if len(line)>8:
            file_target.write(line)
    file_source.close()
file_target.close()



