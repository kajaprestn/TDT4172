# Introduksjon til maskinlæring

## Forelesning 2: Veiledet læring, data og klassifisering

Maskinlæring går ut på å programmere datamaskiner sånn at de kan lære fra data. 
> A computer program is said to learn from experience *E* with respect to some task *T* and some performance measure *P*, if its performance on *T*, as measured by *P*, improves with experience *E*.

- **Experience E:** dataett består av datapunkter med egenskaper - variabler/features
    - kvalitative: kategorier/diskrete sett
    - kvantitative: numeriske verdier/kontinuerlige sett


- **Task T:** Vi har lyst til å finne ut noe fra dataene
Avhengig av dataene og problemet kan vi velge mellom veiledet, uveiledet og forsterket læring. Har vi et target y per datapunkt, passer det bra å gjøre veiledet læring.
    - Vi ønsker å estimere $y$ for nye $x$, basert på relasjonene representert mellom kjente $x$ og $y \rightarrow$ Vi trenger en modell $f$ som estimerer $p(y|\bold{x})$

- **Modell for $p(y|\bold{x})$?**
    - Den enkleste modellen er lineær regresjon: $z = \sum_{i = 1}^n w_i x_i + b = \bold{wx}+ b$, der vektene $w_i$ forteller oss hvor viktig hver feature $x_i$ i $\bold{x}$ er for utfallet. 

- **Performance measure P:** Hvor godt klarer modellen å estimere $y$? 
    - Tilpass modellen $f$ slik at måloppnåelse øker, basert på data. 


- **Tapsfunksjon:**
    - Cross-entropy loss: $$L(y_{pred}, y) = -y ln \sigma (\bold{wx} + b) - (1 - y) ln (1 - \sigma (\bold{wx} + b))$$
    - Modellen vår er definert av **w** og *b*. Disse omtales som modellens parametre $\theta$ = (**w**, *b* ). Vi vil finne parametrene som minimerer tapet, i gjennomsnitt over alle datapunktene. Dette kalles å trene modellen

- **Gradient descent:**
    - Uten å vite noe om funksjonen, ønsker vi å finne dens minimum. 
    - **Hvilken retning bør vi bevege oss i? I hvilket rom?** Den retningen i parameterrommet (rommet spent ut av $\theta$-verdiene) der tapet $L(y_pred, y)$ minker; **Altså**, vi trenger den deriverte av $L$ med hensyn på $\theta$. 
    - **Hvor store steg skal vi ta i denne retningen?** Dette må vi bestemme, og størrelsen kalles læringsraten $\mu$ til modellen. Høyere LR innebærer større endringer av parametrene $\theta$ for hvert steg. Mens $\theta$ er parametrene som tilpasses for at modellen skal passe til dataene, er LR eksempel på en hyperparameter. Dette er parametre som definerer læringsprosessen og ikke modellen. 
> TAPSFUNKSJONEN MÅ VÆRE DERIVERBAR

- **Treningen:** Dataene som brukes for å trene modellen får ikke brukes til å evaluere den - splitt i train og test-data