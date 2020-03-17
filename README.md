# ML
Laborator
#Laborator 5
* Model BagOfWords: este o metodă de reprezentare a datelor de tip text, bazată pe frecvența de apariție a cuvintelor în cadrul documentelor
➔ algoritmul este alcătuit din 2 pași:
1. definirea unui vocabular prin atribuirea unui id unic fiecărui
cuvânt regăsit în setul de date (setul de antrenare)
2. reprezentarea fiecărui document ca un vector de dimensiune
egală cu lungimea vocabularului, definit astfel:
𝑓𝑒𝑎𝑡𝑢𝑟𝑒𝑠[𝑤𝑜𝑟𝑑_𝑖𝑑𝑥] = 𝑛𝑢𝑚ă𝑟𝑢𝑙 𝑑𝑒 𝑎𝑝𝑎𝑟𝑖ț𝑖𝑖 𝑎𝑙 𝑐𝑢𝑣â𝑛𝑡𝑢𝑙𝑢𝑖 𝑐𝑢 𝑖𝑑 − 𝑢𝑙 𝑤𝑜𝑟𝑑_𝑖𝑑

* Build vocabulary: 
  object = Bow()
  object.build_vocab(training_sentences)
  
* Transforming text into numerical features
  training_features = object.get_features(training_sentences)
  testing_features = object.get_features(test_sentences)
  
* Normalizing numerical features before feeding into the classification model
  norm_data = normalize_data(training_features,testing_features,'l2')
  
* SVM model training
  obj = svm.SVC(1,kernel = 'linear')
  obj.fit(norm_data[0],training_labels)
  
* Getting predictions
  predictions = obj.predict(norm_data[1])
  
* Printing accuracy
  accuracy = accuracy_score(test_labels,predictions)
  
* Printing F1-Score
  f1_scor = f1_score(test_labels,predictions,average =None)
