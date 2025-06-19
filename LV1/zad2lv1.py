ocjena = -1
try:
    while ocjena > 1.0 or ocjena < 0.0:
        ocjena = input("Unesi ocjenu ")
        ocjena = float(ocjena)
except:
     print("nisi unio broj!")
if ocjena >= 0.9:
    print("A")
elif ocjena >= 0.8:
    print("B")
elif ocjena >= 0.7:
    print("C")
elif ocjena >= 0.6:
    print("D")
elif ocjena < 0.6:
    print("F")
