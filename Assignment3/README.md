### Dataset

The "spam_classifier.py" script is designed to train on the "emails.csv" file. If you intend to train it on a different dataset(which is not recommended as the hyper parameters are tuned for this dataset), ensure to perform the following steps:

1. Rename the dataset file to "emails.csv".
2. Modify the headers of the CSV file to "text" and "spam". The "text" column should contain the email content, while the "spam" column should contain a binary value indicating whether the email is spam (1) or not spam (0).

## Testing
The emails to be tested should be placed within a folder named "test", located in the same directory as the script file. The emails should be named following this format: "email1.txt", "email2.txt", and so on. Each filename should adhere to the pattern "email{n}.txt", where "n" represents a natural number, and the numbers should be consecutive.

## Output
The predicted outputs (1/0) for the emails in the test directory will be printed individually in the output stream, each on its own line. Additionally, the predictions made by each algorithm—Naive Bayes, SVM, and Logistic Regression—will be written to separate files named "predictions.txt".

### Note
The initial run of the file will take longer because it involves training the model. However, subsequent runs will be faster since the script utilizes pre-trained models, resulting in quicker output generation.