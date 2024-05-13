### Dataset
The spam_classifier.py script trains on emails.csv file. If you wish to train it on a different dataset, then make sure to do the following:

1. Rename the file to emails.csv
2. Change the headers of the .csv file to to "text", "spam" where the text column contains the mail and the spam column contains 1 if it is a spam else 1.

## Testing
The mails that needs to be tested should be put under a folder named "test" and should be named as email1.txt, email2.txt, .... The general format is email{n}.txt where n is a natural number and should be continuous.

## Output
The predicted outputs (1/0) for the emails in the test directory will be printed individually in the output stream, each on its own line. Additionally, the predictions made by each algorithm—Naive Bayes, SVM, and Logistic Regression—will be written to separate files named "predictions.txt".