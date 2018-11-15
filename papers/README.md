#Sorodni članki
Kandogan, Eser. "Just-in-time annotation of clusters, outliers, and trends in point-based data visualizations." Visual Analytics Science and Technology (VAST), 2012 IEEE Conference on. IEEE, 2012.

(**Motivacija**) 
Kandogan v članku predlaga metodo za identifikacijo in anotacijo gruč, osamelcev znotraj gruč in trendov na točkovnih vizualizacijah. Trend v tem kontekstu predstavlja atribut v originalnem prostoru ter smer na vizualizaciji, v kateri ta atribut znotraj gruče narašča.
(**Metode**) 
Za identifikacijo gruč avtor uporabi precej preprost in hiter pristop, ki temelji na mrežni razdelitvi projekcije in združevanju gostih celic. Osamelci znotraj gruč so identificirani na podlagi razdalj (v standardnih odklonih) njihovih vrednosti atributov od njihovega povprečje v gruči. Trendi znotraj gruč so identificirani s pomočjo linearne regresije. Z identificiranimi značilkami se nato za vsak atribut v originalnem prostoru izračunajo razne metrike s pomočjo katerih izberejo zanimive atribute. Primeri teh metrik so moč trendov tega atributa, varianca znotraj gruč in prekrivanje vrednosti atributa med gručami. Za vsakega izmed izbranih atributov in gruč se nato zgenerirajo možne anotacije z najpogostejšimi vrednostmi (diskretni atributi), intervali (zvezni atributi) in posebne anotacije za trende.Uporabnik lahko nato izbere enega izmed zanimivih atributov (ali kombinacijo večih) ter način prikaza (npr. pričakovane vrednosti). Glede na izbrano se izrišejo ustrezne anotacije.
(**Vrednotenje**)
Metoda je ovrednotena z natančnostjo in preklicem zgeneriranih anotacij za eno podatkovno množico. Natančnost za anotacijo je izračunana kot delež točk v gruči, za katere ta anotacija drži, preklic pa kot razmerje točk v gruči za katere ta anotacija drži in vseh točk za katere ta anotacija drži. Poleg tega so predstavili tudi primera uporabe na dveh podatkovnih množicah.
(**Povezanost**)
Delo predstavlja zanimiv pristop za anotacijo točkovnih vizualizacij. Ta pristop omejuje to, da za vse gruče na vizualizaciji prikaže enake načine prikaza in atribute (izbrane s seznama globalno zanimivih), namesto najbolj zanimivih za vsako gručo posebej, kar bomo poiskusili doseči v svojem delu. 

Heimerl, Florian, et al. "DocuCompass: Effective exploration of document landscapes." Visual Analytics Science and Technology (VAST), 2016 IEEE Conference on. IEEE, 2016.

(**Motivacija**) 
Heimerl et. al. predlagajo interaktivno metodo, ki omogoča raziskovanje projekcij dokumentov
na nivoju bližnjih podskupin z uporabo leče.
(**Metode**)
V osnovi ta vizualizacija prikaže seznam ključnih besed za dokumente, ki so znotraj leče. Če je katera izmed pomembnih besed označena, se na projekciji označijo vsi 
dokumenti, ki vsebujejo to besedo. Za izbiro pomembnih besed omenijo več pristopov, predlagajo pa uporabo mere G^2, ki primerja ponovitve posameznih besed
v dokumentih pod lupo s ponovitvami v dokumentih izven lupe. Vizualizacija podpira tudi lokalno gručenje, ki točke v lupi in njeni bližnji okolici
razvrsti v gruče v originalnem prostoru ter točke na projekciji obarva glede na razvrstitev. Ta funkcija naj bi uporabniku pomagala pri izbiri velikosti lupe, poleg tega pa lahko razkrije tudi
napake v projekciji. 
(**Vrednotenje**)
Pristop so ovrednotili s študijo, kjer so kandidati reševali naloge s tem pristopom in s pristopom, ki temelji na raziskovanju
posameznih dokumentov. Vsi udeleženci so izbrali pregledovanje z lečo kot boljši pristop.
(**Povezanost**)
Metoda je zelo podobna temu, kar bomo za dinamično raziskovanje projekcij poizkusili mi. Razlika je seveda, da je ta omejena
na tekstovne podatke. V našem primeru bomo namesto besed iskali pomembne vrednosti atributov.

da Silva, R., et al. "Attribute-based visual explanation of multidimensional projections." Proc. EuroVA (2015): 134-139.

(**Motivacija**) 
Silva et. al. v tem delu predlagajo metodo za obarvanje točkovnih projekcij s ciljem, da nam v jeziku originalnih atributov obrazloži sosednosti bližnjih točk.
(**Metode**)
Metoda obarva vsako točko glede na to kateri atribut najbolj prispeva k podobnosti s sosednimi točkami. Vmesni piksli so interpolirani.
Sosednost točk je definirana z radijem. Predlagani sta dve meri za računanje prispevka atributa k podobnosti s sosednimi točkami: razmerje lokalnega (sosedi) in globalnega (vse točke) 
prispevka atributa k evklidski razdalji ter razmerje lokalne in globalne variance atributa.
Izračuna se tudi zaupanje v te pomembnosti, ki je na vizualizaciji predstavljeno kot svetlost. Izračun zaupanja temelji na ujemanju pomembnosti atributov med sosednimi točkami.
Za boljšo preglednost izvedejo tudi filtriranje atributov kjer ohranijo le atribute, ki so pomembni za veliko število projiciranih točk.
(**Vrednotenje**)
Metodo so demonstrirali z obrazložitvijo projekcij na treh resničnih podatkovnih množicah.
(**Povezanost**)
Predlagana vizualizacija je še en možen pristop statične razlage podatkovnih projekcij. Statičen del pristopa, ki ga bomo razvili mi 
bo temeljil na dobri razdelitvi projekcije v gruče. Omenjen pristop pa je precej koristen ravno takrat, ko take razdelitve nimamo.

Heulot, Nicolas, Michael Aupetit, and Jean-Daniel Fekete. "Proxilens: Interactive exploration of high-dimensional data using projections." VAMP: EuroVis Workshop on Visual Analytics using Multidimensional Projections. The Eurographics Association, 2013.

(**Motivacija**) 
Heulot et. al. predlagajo
pristop, ki nam pri interpretaciji visokodimenzionalnih projekcij pomaga upoštevati napake le teh skozi interaktivno raziskovanje z uporabo leče.
(**Metode**)
Metoda nam omogoča, da se z lupo osredotočimo na specifično točko v projekciji. Ostale točke, ki so znotraj lupe se glede na razdaljo do te točke v originalnem prostoru in določen prag razdelijo na dve skupini.
Prva skupina vsebuje točke, ki v originalnem prostoru niso blizu osredotočene točke. Te točke se premaknejo na rob lupe in obarvajo drugače. Druga skupina točk ostane na istem mestu, področje okoli njih pa se obarva
glede na razdaljo do osredotočene točke v originalnem prostoru. 
Ta pristop označi tudi točke, ki niso v lupi ampak so v originalnem prostoru bližje kot določen prag.
(**Vrednotenje**)
Metodo so demonstrirali na dveh umetnih podatkovnih množicah in na podatkovni množici razpoznave rokopisnih števil, kjer se je izkazala kot dober pristop
za analizo osamelcev in za pregledovanje strukture razrednih skupin.
(**Povezanost**)
Predlagana metoda je zanimiv primer dinamičnega raziskovanja točkovnih projekcij. Poleg tega je redek pristop, ki koristi lupo za
vsesplošne projekcije visoko dimenzionalnih podatkov. Določene dele tega pristopa bi lahko vključili tudi v naš pristop z lupo. 

Stahnke, Julian, et al. "Probing projections: Interaction techniques for interpreting arrangements and errors of dimensionality reductions." IEEE transactions on visualization and computer graphics 22.1 (2016): 629-638.

(**Motivacija**) 
V tem delu Stahnke et. al. predlagajo orodje za interaktivno raziskovanje projekcij visoko dimenzionalnih podatkov.
(**Metode**)
Orodje podpira raziskovanje na nivoju posameznih točk in na nivoju skupin. Če je na projekciji izbrana ena podatkovna točka se za vsak atribut
v originalnem prostoru izpiše njegova vrednost in prikaz pozicije te vrednosti na distribuciji atributa. Skupine točk lahko pridobimo z ročno izbiro, hierarhičnim razvrščanjem ali pa na podlagi razrednega atributa. Izbira
dveh skupin prikaže primerjavo povprečij ter porazdelitev za vse atribute. Orodje podpira tudi raziskovanje vpliva
atributov na projekcijo. Če iz seznama atributov izberemo enega se ozadje projekcije obarva z dvodimenzionalno porazdelitvijo atributa v projekciji.
Predstavljeni sta tudi dve tehniki za analizo napak projekcij. Prva okoli vsake točke na projekciji izriše obroč, katerega radij je odvisen od kumulativne
napake projekcije za to točko. Barva obroča indicira ali je večino primerov v originalnem prostoru bližje ali dlje kot prikazuje projekcija. Drugi način
nam omogoča, da se osredotočimo na specifično točko, ob čem se projekcije ostalih točk popravijo glede na razdaljo do dane točke v prvotnem prostoru. 
Poti popravkov so uporabniku jasno prikazane.
(**Vrednotenje**)
Predlagane tehnike so ovrednotili s študijo pri kateri so udeleženci dobili razne naloge na podatkovni množici 36 držav in osem socioekonomskih atributov.
Večina udeležencev je bila sposobnih rešiti naloge samostojno ali pa z nekaj pomoči.
(**Povezanost**)
Predstavljeno orodje je primer dinamičnega in interaktivnega raziskovanja projekcij. V primerjavi s cilji našega dela vključuje veliko ročnega dela
in informacij ne filtrira avtomatsko glede na njihovo pomembnost, kar lahko z večanjem števila atributov predstavlja precejšnji problem. 
