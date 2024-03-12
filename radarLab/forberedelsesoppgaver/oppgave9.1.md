## Forberedelsesoppgave 9.1

1. Beregn teoretisk dopplerskift som funksjon av radiell hastighet ved 24.13 GHz senterfrekvens.
    * Vi bruker formelen $f_D=\frac{2f_0 v_r}{c}$
    * $f_D \approx 160.9 v_r$
    * $v_r \approx f_D/160.9$
    * $δfD = 1/T$
    * Der $v_r$ er bilen sin hastighet i ønsket retning, og T er observasjonstiden (dvs tiden radaren observerer målet mens det er i bevegelse). Dette vil i
praksis være måletiden (lengden på måledatane dere samler inn). Hvis hastigheten er konstant under
observasjonstiden, vil δfD tilsvare 3dB-bredden på frekvenslinjen i dopplerspekteret som kommer
fram etter Fouriertransformen
2. Beregn antennevinning ut fra ligning (III.7) og sammenlign med data som ﬁns i databladet
for radaren.
    * Vi bruker formelen $G[dBi] \approx 10 log_{10}(\frac{3*10^4}{\theta_e \theta_a})$
    * Vi må bruke $\theta \approx arctan(\frac{\lambda}{D})$ for hhv høyde og bredde på radaren for å beregne hhv $\theta_e$ og $\theta_a$
    * $\theta \approx arctan(\frac{\lambda}{D})$ (1.0)
    * $\lambda = c/f \approx \frac{3*10^8}{24 * 10^9} = 0.0125 m$
    * Using the $\theta$ equation with both the heigth $D_e = 65.4 mm$ and width $D_a = 25mm$ of the radar respectivly, we get: $\theta_e = 10.82^{\circ}$ and $\theta_a = 26.57^{\circ}$

3. Beregn radartverrsnittet ved 24 GHz til en hjørnereﬂektor som har sidekant a = 21 cm.
   * Vi bruker at $A_{eff}=\frac{a^2}{\sqrt{3}}$ og setter inn for a.
   * Da får vi $A_{eff}=0.025m^2$
   * For å finne det effektive radartverrsnittet bruker vi at $\sigma = \frac{4\pi a^4}{3\lambda^2} = 52.1$

4. Hvor mye må reﬂektoren beveges radielt for at I-Q-phasoren skal foreta et 360-graders fase-omløp?
    * Man skulle tro det var en bølgelengde man måtte bevege seg en hel bølgelengde. Men radarsignalet blir jo reflektert tilbake. Dermed er det en halv bølgelengde altså $\lambda/2 = 0.0125m/2 = 0.00625m = 6.25mm$