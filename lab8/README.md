CERINTA 2. Antrenati un Perceptron cu algoritmul Widrow-Hoff pe multimea de antrenare
de la exercitiul anterior timp de 70 epoci cu rata de invatare 0.1. Care este
acuratetea pe multimea de antrenare? Apelati functia plot_decision_boundary la
fiecare pas al algoritmului pentru a afisa dreapta de decizie.
     ➔  Algoritmul Widrow-Hoff, numit si metoda celor mai mici patrate (Least mean
squares), este un algoritm de optimizare a erorii perceptronului pe baza metodei
coborarii pe gradient tinand cont doar de eroare de la exemplul curent.
Regula de actualizare foloseste derivata partiala a functiei de pierdere, in functie de
ponderi si bias. In continuare vom calcula detaliat derivatele partiale ale functiei de
pierdere. Functia de activare a perceptronului din algoritmul Widrow-Hoff este
identitatea (f(x) = x ).
     ➔ def train_perceptron(X,Y, epochs, lr):

CERINTA 3. Antrenati un Perceptron cu algoritmul Widrow-Hoff pe multimea de antrenare
X =[ [0, 0], [0, 1], [1, 0], [1, 1] ], y = [-1, 1, 1, -1]. Care este acuratetea pe
multimea de antrenare? Apelati functia plot_decision_boundary la fiecare pas al
algoritmului pentru a afisa dreapta de decizie.


CERINTA 4. Antrenati o retea neuronala pentru rezolvarea problemei XOR cu arhitectura
retelei descrise in 3, si algoritmul coborarii pe gradient descris in 4, folosind
70 epoci, rata de invatare 0.5, media si deviatia standard pentru initializarea
ponderilor 0, respectiv 1, si 5 neuroni pe stratul ascuns. Afisati valoarea
erorii si a acuratetii la fiecare epoca. Apelati functia plot_decision la fiecare pas
al algoritmului pentru a afisa functia de decizie.

  ➔ Algoritmul coborarii pe gradient se bazeaza pe derivata de ordinul 1, pentru a gasi
minimul functiei de pierdere. Pentru a gasi un minim local al functiei de pierdere, vom
actualiza ponderile retelei proportional cu negativul gradientului functiei la pasul
curent.
In continuare vom detalia implementarea (pseudo-cod) algoritmului de coborare
pe gradient pentru reteaua descrisa anterior.
                   Pasii algoritmului sunt:

     ➔ PAS1) Initializare ponderilor - ponderile si bias-ul retelei se initializeaza aleator cu
valori mici aproape de 0 sau cu valoare 0.

      ➔ PAS2) Pasul forward - Vom defini o metoda forward care calculeaza predictia retelei
folosind ponderile actuale si datele de intrare date ca parametri, apoi vom
calcula pentru fiecare strat valoarea lui z (z = inmultirea datelor de intrare cu
ponderile si adunarea bias-ului) si valoarea lui a (a = aplicarea functiei de
activare lui z, ( a = f(z) )).

             ➔ def fwd(x, W_1, W_2, b_1, b_2):

      
     ➔ PAS3) Calculam valoarea functiei de eroare (logistic loss) si acuratetea
    
             ➔ loss = (-y .* log(a_2) - (1 - y) .* log(1 - a_2)).mean()
             ➔  accuracy_4 = (round(a_2) == y_4).mean()

     ➔PAS4) Pasul backward - vom defini o metoda backward care calculeaza derivata
functiei de eroare pe directiile ponderilor, respectiv a fiecarui bias. Vom
incepe calculul cu derivata functiei de pierdere pe directia z_2 folosind regula
de inlantuire (chain-rule) a derivatelor.

              ➔ def backward(a_1, a_2, z_1, W_2, X, Y, num_samples):

      ➔ PAS5) Actualizarea ponderilor - ponderile se actualizeaza proportional cu negativul
mediei derivatelor din batch (mini-batch).

       ➔ PAS6) Pentru a antrena o retea neuronala cu ajutorul algoritmului coborarii pe
gradient trebuie sa:
                  a) Stabilim numarul de epoci
                  b) Stabilim rata de invatare
                  c) Sa initiliazam ponderile (pasul 1)
                  d) Sa amestecam datele la fiecare epoca
                  e) Sa luam un subset din multimea (sau toata multimea) de antrenare si
                  sa executam pasii 2, 3, 4, 5 pana la convergenta
