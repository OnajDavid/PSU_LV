song_words = {}
fsong = open('C:\\Users\\Korisnik\\Downloads\\song.txt')
for line in fsong:
    line = line.rstrip()
    words = line.split()
    for word in words:
        if word in song_words:
            song_words[word] += 1
        else:
            song_words[word] = 1
counter = 0
for key,value in song_words.items():
    if value == 1:
        counter += 1
        print(key)
print(counter)
fsong.close()