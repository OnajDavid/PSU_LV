def total_euro(sati,placa):
    rez = sati * placa
    return rez

radni_sati = input()
satnica = input()
radni_sati = int(radni_sati)
satnica = int(satnica)
print("Ukupno",total_euro(radni_sati,satnica))


