1.  Normalizam datele pentru setul de date Car Price Prediction: def normalize(train, test){...}
2.  Car Price Prediction este impartit in 3 fold-uri, si antrenam un model de regresie liniară 
      *linear_regression_model = LinearRegression()
3.  Calculam valoarea medie a funcțiilor MSE și MAE. 
4.  Antrenam la fel un model de regresie ridge.
      *ridge_regression_model = Ridge()
5. Calculam din nou valoarea medie a funcțiilor MSE și MAE.
