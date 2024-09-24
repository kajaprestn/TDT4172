# Introduksjon til maskinlæring

## Forelesning 3: Klassifisering Bayes’ theorem, ubalanserte data og beslutningstrær

### Bayes' teroem i klassifiseringssammenheng
Vi ønsker å estimere sannsynligheten for en klasse for et datapunkt, altså $p(C_k | x)$. Fra Bayes' teorem har vi 
$$p(C_k | x) = \frac{p(x | C_k)p(C_k)}{p(x)}$$

der

$p(C_k | x)$ er sannsynlighet for klasse (hypotesen) i lys av data. Oppdatert prior. (*posterior*)

$p(x | C_k)$ er sannsynlighet for å observere dataene x gitt en klasse (gitt at hypotesen stemmer) (*likelihood*)

$p(C_k)$ er sannsynligheten for klassen (hypotesen), uten ytterligere informasjon (*prior probability*)

$p(x)$ er sannsynlighetsfordelingen av dataene (*evidence*)

*posterior* er NØYAKTIG det vi ønsker å estimere ved hjelp av ML $\rightarrow$ Har vi alle faktorene i BT, er ikke ML nødvendig.

I klassifiseringsoppgaver er det ikke nødvendig å beregne sannsynligheten for dataene $p(x)$ eksplisitt fordi den er uavhengig av klassen $C_k$. Siden vi bare ønsker å sammenligne sannsynlighetene for forskjellige klasser og (og $p(x)$ er konstant for alle klasser), kan vi utelate p(x) fra beregningen og formelen blir

$$p(C_k | x) \propto p(x | C_k)p(C_k)$$


### Naïv Bayes

I Naive Bayes antar vi at alle features, \( x_i \), er betinget uavhengige av hverandre gitt klassen, \( C_k \). Dette gjør at vi kan faktorere sannsynligheten for å observere alle features som produktet av sannsynlighetene for hver enkelt feature: 

$$p(x | C_k) = \prod_{i=1}^n p(x_i | C_k)$$
 
Denne antagelsen forenkler beregningen av *likelihood*, selv om den ikke alltid er realistisk. Selv om uavhengighetsantagelsen forenkler beregningen av likelihood, er den ofte en forenkling av virkeligheten. I mange praktiske tilfeller vil det være avhengigheter mellom funksjonene, men til tross for denne forenklingen fungerer Naive Bayes overraskende godt i mange praktiske applikasjoner.

I praksis velger Naive Bayes den klassen, \( C_k \), som maksimerer det totale produktet av sannsynligheten for klassen \( p(C_k) \) og sannsynligheten for hvert feature \( p(x_i | C_k) \). Dette betyr at vi sammenligner sannsynlighetene for hver klasse og velger den med høyest verdi for det gitte datapunktet.

$$p(C_k | x) \propto p(C_k) \prod_{i= 1}^n p(x_i | C_k)$$

Vi kan skrive om likelihooden for data slik hvis vi antar at alle features er uavhengige av hverandre. 

$$\hat{y} = \argmax_{k \in \{1, \dots, n\}} p(C_k) \prod_{i=1}^{n} p(x_i | C_k)$$

der

\(\hat{y}\) er estimat av klasse.


#### Gaussisk fordeling

Den Gaussiske fordelingen (normalfordelingen) er en sannsynlighetsfordeling som brukes til å modellere mange naturlige fenomener. Funksjonen \( F(x) \) for den Gaussiske fordelingen er gitt som:

$$F(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2 \sigma^2}}$$

der:

- \( \mu \) er forventningsverdien til \( F(x) \) (midtpunktet på kurven, også kjent som gjennomsnittet)
- \( \sigma \) er standardavviket til \( F(x) \), som beskriver spredningen av dataene rundt gjennomsnittet (varians = \( \sigma^2 \))

Den Gaussiske fordelingen er symmetrisk rundt forventningsverdien \( \mu \), og sannsynligheten for verdier lenger unna \( \mu \) avtar eksponentielt.

#### Begrensninger og praktisk bruk

Den Gaussiske fordelingen er mye brukt i statistikk og maskinlæring fordi mange naturlige fenomener følger en tilnærmet normalfordeling når prøvestørrelsen er stor (sentralgrenseteoremet). Men det er viktig å merke seg at ikke alle datasett følger en normalfordeling, spesielt når dataene har skjevheter, uteliggere, eller ikke har en symmetrisk fordeling rundt et gjennomsnitt.



#### Er det mulig å berege ikke-naïv Bayes? Altså den *faktiske* klassen?

$$p(C_k | x) = \frac{p(x | C_k)p(C_k)}{p(x)}$$

der

$p(x | C_k)p(C_k)$ er *likelihood*, sannsynligheten for $x$ gitt en klasse $\rightarrow$ Denne kan vi beregne om vi samler data som representerer alle kombinasjoner av alle featureverdier for alle klasser. Det er ~umulig, særlig om vi går bort fra antakelsen om uavhengige features. 

$p(x)$ er *evidence*, sannsynlighetsfordelingen til dataene $\rightarrow$ Denne er vanskelig. Da må vi kjenne simultanfordelingen til alle dataens features. Men: For klassifisering påvirker den fremdeles sannsynligheten for alle klasser likt

I de aller, aller fleste tilfeller som er nyttige for den virkelige verden: **NEI.**

Og nå skjønner vi hvorfor vi driver med maskinlæring: Fordi vi ikke har noe valg

Hadde vi klart å finne et uttrykk for $p(y|x)$ for problemene våre, hadde vi brukt det. 


### Ubalanserte data
Oppstår når fordellingen av klasser ikke er jevn i et datasett. Set betyr at én eller flere klasser har betydelig flere datapunkter enn andre klasser. 

#### Hva kan vi gjøre?
**Fattigmannsløsningen** er å endre terskelen mellom klassene for prediksjonen:

```pred_to_class = [1 if _y > threshold else 0 for _y in y_pred]```

Dette kan gjøres f.eks. basert på precision-recall-plottet, eller avhengig av metrikken man vil maksimere. Ved å senke terskelen kan vi øke andelen predicted positive, fordi alt over terskelen predikeres som 1. Å maksimere recall/sensitivitet går som oftest på bekostning av andre metrikker (accuracy, presisjon)

Ellers kan man

#### Justere dataene

Vi kan gjøre *resampling*. Her har vi to valg

- **Undersampling:** Vi bruker så mange datapunkter fra den overrepresenterte klassen som vi har tilgjengelig i den underrepresenterte klassen. Da ender vi opp med like mange datapunkter fra hver klasse, men potensielt veldig få datapunkter totalt. Dette kan føre til **undertilpasning (under-fit)**.
- **Oversampling:** Vi kopierer instanser fra den underrepresenterte klassen, slik at vi får like mange totalt som av den overrepresenterte klassen. Da ender vi også opp med like mange datapunkter fra hver klasse, men potensielt mange duplikater. Dette kan fører til **overtilpasning (over-fit)**. 
- **Ubalanserte data**
    - **Oversampling** kan gjøre at modellen **overtilpasser** seg til den kopierte klassen $\Rightarrow$ Modellen memorerer de kopierte punktene. Dette er også kjent som **høy varians**
    - **Undersampling** kan føre til at informasjon fra den overrepresenterte klassen går tapt $\Rightarrow$ Modellen boommer på punkter fra denne klassen. Dette er også kjent som **høy bias**

Eller

#### Justere tapsfunksjonen
 
I stedet for å endre hvilke data modellen får tilgang til, kan vi fortelle modellen at en av klassene (ofte den underrepresenterte) er ekstra viktig 


### Beslutningstrær

- Terminologi:
    - Rotnode - starten på treet, der alle features kommer inn. Har splitting criteria
    - Løvnoder er alle noder under rotnoden som har ett eller flere splitting critieria
    - Beslutningsnoder er noder som ikke splitter dataene. De inneholder predikert verdi for datainnstansen som ble send gjennom treet
    - Trestumper - består av én rotnode og n beslutningsnoder (for n mulige utfall; i vårt tilfelle to), og splitter dataene på én feature
- Tenk over:
    - Hvordan velger vi **rekkefølge å splitte på** hvis vi ikke vet nok om features (feks hvis de beskriver pingviner)?
    - Når skal/må vi **slutte å splitte**, selv om nederste node har instanser fra begge klassene?
    - Hva gjør vi med eventuelle beslutningsnoder som har **instanser fra begge klassene**?

### Velge rekkefølge å splitte på
Vi ønsker å splitte på en feature som lar oss ta beslutnignen tidligst mulig, altså reduserer usikkerheten mest mulig. 
Redusert usikkerhet: usikkerhet før splitt minus usikkerhet etter splitt
Hvordan måler vi denne usikkerheten? F.eks. *Gini Impurity*

#### Gini Impurity
Et tall i [0, 0.5] som angir sannsynligheten for at nye, tilfeldige data feilklassifiseres hvis det gis et tilfeldig klasse-label i henhold til klassedistribusjonen i datasettet. 

#### Entropi
Gitt en sansynlighetsfordeling over $k$ klasser, er sannsynligheten for hver klasse $p_i$. Entropien til fordelingen er 
$$I(p_1, ..., p_k) = - \sum_{i = 1}^{k} p_i log_2 (p_i)$$

Vi bør splitte på den featuren som gir **størst** endring av entropi.

**Før** treet spliiter på en feature har vi $p$ positive og n negative instanser i dataene. Dette svarer til en binær distribusjon der positiv klasse har sannsynlighet p / (p + n), og negativ klasse har sannsynlighet n / (p + n):


### Når har man laget en løvnode? 
Vi har åpenbart laget en beslutningsnode hvis alle instansene i noden tilhører samme klasse.

Når slutter vi å splitte, selv om nederste node har instanser fra begge klassene?

1. Når vi ikke har flere features igjen å teste. Det kan skje at dataene ikke gjør det mulig å splitte perfekt. 
2. Når vi ikke har flere eksempler igjen. Før eller siden har alle kombinasjoner av features som er tilgjengelig i dataene blitt testen, uten at alle mulige kombinasjoner av features er testet

Hva gjør vi med eventuelle noder som har instanser fra begge klassene? 

Det vanligste er å returnere den dominante klassen i noden, altså label til klassen med flest instanser i noden. 

En annen mulighet er å gjøre en tilfeldig trekning av label fra treningsdataene, altså å predikere a priori-sannsynligheten.