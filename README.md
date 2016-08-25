# Naive-Bayes for Elefriends Post Classification

Mega-simple Naive Bayes classification of posts, using NLTK's built in
NaiveBayesClassifier.

Features are binary word occurrences.  Posts are tokenized and lemmatized, and stopwords are removed, all using standard NLTK/wordnet modules.

Performance in 10-fold cross validation classifying posts flagged with
self-harm/suicide content vs. unflagged posts:

```
Fold 0
	Accuracy = 0.9067796610169492
	Precision = 0.4901185770750988
	Recall = 0.6424870466321243
	F-score = 0.5560538116591929
	Type 1 err = 0.5098814229249012
	Type 2 err = 0.03687867450561197

##########

Fold 1
	Accuracy = 0.8865348399246704
	Precision = 0.42356687898089174
	Recall = 0.689119170984456
	F-score = 0.52465483234714
	Type 1 err = 0.5764331210191083
	Type 2 err = 0.03314917127071823

##########

Fold 2
	Accuracy = 0.8945386064030132
	Precision = 0.45110410094637227
	Recall = 0.7409326424870466
	F-score = 0.5607843137254902
	Type 1 err = 0.5488958990536278
	Type 2 err = 0.02767017155506364

##########

Fold 3
	Accuracy = 0.880357983984927
	Precision = 0.4109195402298851
	Recall = 0.7447916666666666
	F-score = 0.5296296296296297
	Type 1 err = 0.5890804597701149
	Type 2 err = 0.0276056338028169

##########

Fold 4
	Accuracy = 0.8813000471031559
	Precision = 0.4166666666666667
	Recall = 0.78125
	F-score = 0.5434782608695653
	Type 1 err = 0.5833333333333334
	Type 2 err = 0.0238230289279637

##########

Fold 5
	Accuracy = 0.8492699010833726
	Precision = 0.3476190476190476
	Recall = 0.7604166666666666
	F-score = 0.47712418300653586
	Type 1 err = 0.6523809523809524
	Type 2 err = 0.02701115678214915

##########

Fold 6
	Accuracy = 0.83843617522374
	Precision = 0.3295711060948081
	Recall = 0.7604166666666666
	F-score = 0.4598425196850394
	Type 1 err = 0.6704288939051919
	Type 2 err = 0.02738095238095238

##########

Fold 7
	Accuracy = 0.8704663212435233
	Precision = 0.38108882521489973
	Recall = 0.6927083333333334
	F-score = 0.4916820702402958
	Type 1 err = 0.6189111747851003
	Type 2 err = 0.03325817361894025

##########

Fold 8
	Accuracy = 0.8657560056523788
	Precision = 0.3798449612403101
	Recall = 0.765625
	F-score = 0.5077720207253886
	Type 1 err = 0.6201550387596899
	Type 2 err = 0.025921658986175114

##########

Fold 9
	Accuracy = 0.8496701225259189
	Precision = 0.3432098765432099
	Recall = 0.7239583333333334
	F-score = 0.4656616415410385
	Type 1 err = 0.6567901234567901
	Type 2 err = 0.030867792661619105
```
