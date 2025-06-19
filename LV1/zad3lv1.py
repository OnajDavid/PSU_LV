brojevi = []
num = ""
while num != "Done":
    num = input()
    brojevi.append(num)
brojevi.pop()
print(len(brojevi))
print(brojevi)
average = 0.0
for num in brojevi:
    average += float(num)
average /= len(brojevi)
print(average)
print(max(brojevi))
print(min(brojevi))
print(brojevi.sort())

        