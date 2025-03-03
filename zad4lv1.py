flocation = "C:\\Users\\student\\Desktop\\LV SU\\"
datoteka = input()
flocation += datoteka
file = open(flocation)
average = 0.0
count = 0
for line in file:
    line = line.rstrip()
    if line.startswith("X-DSPAM-Confidence:"):
       count += 1
       print(line)
       lsplit = line.split(": ")
       num = lsplit[1]
       average += float(num)
average /= float(count)
print("Average is",average)
file.close()

