file = open('C:\\Users\\student\\Desktop\\LV SU\\song.txt')
songw ={}
for line in file:
    line = line.rstrip()
    words = line.split(" ")
    for word in words:
        if word in songw.keys():
            songw[word] += 1
        else:
            songw[word] = 1
print(songw)

file.close()