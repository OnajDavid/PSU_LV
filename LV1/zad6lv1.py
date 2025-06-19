sum_ham = 0
count_ham = 0
sum_spam = 0
count_spam = 0
spam_exclamation_mark = 0
fspam = open('C:\\Users\\Korisnik\\Downloads\\SMSSpamCollection.txt')
for line in fspam:
    line = line.rstrip()
    words = line.split()
    if words[0] == "ham":
        sum_ham += 1
        for i in words:
            count_ham += 1
    elif words[0] == "spam":
        sum_spam += 1
        for i in words:
            count_spam += 1
        word = words[-1]
        if word[-1] == '!':
            spam_exclamation_mark += 1
print("average words of non-spam: ", count_ham/sum_ham)
print("average words of spam: ", count_spam/sum_spam)
print("number of times '!' was the last letter in spam: ", spam_exclamation_mark)
fspam.close()