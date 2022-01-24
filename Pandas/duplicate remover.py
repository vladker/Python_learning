file_source = open("C:/Users/rogac/Downloads/DICS/totality.dic", mode="r", encoding="latin-1")

line_set=set()
for line_base in file_source:
    line_set.add(line_base)
file_source.close()

file_target = open("C:/Users/rogac/Downloads/DICS/totality2.dic", mode="a+", encoding="latin-1")
for elements in line_set:
    file_target.write(elements)
file_target.close()
