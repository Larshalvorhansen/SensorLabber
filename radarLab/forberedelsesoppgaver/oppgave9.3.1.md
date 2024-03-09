### 9.3.1
* Hvorfor er disse knekkfrekvensene
fornuftige å bruke bl.a ut fra kjennskap til forventet dopplerskift og AD-konverterne som brukes?
    * Knekkfrekvensene som gir opphav til båndbredden $3Hz - 48GHz$ er fornuftige fordi de fleste objekter som måles med 24GHz radar resulterer i et frekvenspekter mellom $3Hz - 24GHz$. For å unngå aliasing setter vi øvre grense på Nyquist frekvensen som er $2*24GHz = 48GHz$.
  * I tilleg ligger det en DC komponent å 


Finn ut hvilke komponenter i oppkoplingen til op-amp/ﬁlter som bestemmer henholdsvis
øvre- og nedre knekkfrekvens og beregn disse komponentverdiene ut fra de spesiﬁserte knekk-
frekvensene. Mål frekvensresponsen til forsterken med ﬁlteret. Lagre data slik at dere kan
generere Bode-plott til labrapport.
1. Bruk radaren til å måle hastigheten til et objekt som beveger seg radielt i mot eller vekk fra
radaren. Finn en egnet alternativ metode for å måle hastigheten (f.eks vha stoppeklokke over
en kjent avstand). Bruk forskjellige hastigheter og lag plott av målt hastighet versus teoretisk
hastighet. Bruk en kompleks Fouriertransform av I- og Q-signalene til å ﬁnne spekteret slik
at en kan ﬁnne både negativt og positivt dopplerskift.