Title : Webtoon Content Classifier

Overview :
This project is a machine learning classifier that categorizes webtoon descriptions into different genres, such as ROMANCE, ACTION, and FANTASY. Inspired by webtoon platforms, the project demonstrates how text classification can be applied to predict the genre of a webtoon based on its description.
We use Python and scikit-learn to implement the classifier, which is trained on a small dataset of 10-15 webtoon descriptions. The model can predict the genre for new webtoon descriptions using a decision tree classifier.

Features :
- Classifies webtoon descriptions into genres like Romance, Action, and Fantasy.
- Uses a decision tree classifier for predictions.
- Handles a small dataset of manually curated webtoon descriptions.

Dataset :
The dataset consists of 10-15 webtoon descriptions, manually categorized into genres. This serves as a training set for the model.

Technologies Used :
- Python: For scripting and building the classifier.
- scikit-learn: For implementing the machine learning model.
- pandas library: For handling and manipulating data.
- TfidfVectorizer: For converting text into numerical features.
- DecisionTreeClassifier: The machine learning model used for classification.

Output :
Accuracy: 0.25
Classification Report:
               precision    recall  f1-score   support

      Action       0.25      1.00      0.40         1
     Fantasy       0.00      0.00      0.00         1
     Romance       0.00      0.00      0.00         2

    accuracy                           0.25         4
   macro avg       0.08      0.33      0.13         4
weighted avg       0.06      0.25      0.10         4
Predicted genre: Action


Installation :
To set up and run the project on your local machine, follow these steps:

Clone the repository:
   ```bash
   git clone https://github.com/vydurya-patil/Webtoon.git
   cd Webtoon 
