# TTT4280 - Sensorer og instrumentering - Optikk Laben


Kode/data til sensor/målesystem i  laboppgaver. 
fft.py inneholder kjerne koden for analysen av data 

record_video_upgrade.py: Samme funksjonalitet som `record_video.py`, men funker uten den eldgamle kameramodulen. Anbefales. Trenger apt-pakken `python3-picamera2`.

read_video_from_roi.py: Henter tidssignalet fra et interessant område i videoopptaket som du velger; skal bli på datamaskinen din. Hvis du har koblet deg opp mot Pi-en med `ssh -Y bruker@pi-en.local`, kan du kjøre den på Pi-en, men det vil gå tregt med grafikken.

simple_model.py: Kodeskjelett til forberedelsesoppgavene; blir på datamaskinen din.
muabd.txt og muabo.txt: Data for hhv deoksygenert og oksygenert blod; «simple_model» er avhengig denne.
raspi-cam-v2.zip: Datablader til kameraet.

forberedelsesoppgaver inneholder alle oppgavene som skulle blitt gjort før og under laben.
