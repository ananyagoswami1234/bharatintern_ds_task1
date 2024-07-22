# bharatintern_ds_task1

Sure, here's a detailed LinkedIn caption for your SMS Classifier Project:



🚀 Project Spotlight: SMS Classifier Using Machine Learning Techniques 🚀

I'm excited to share my latest project where I developed a machine learning model to classify SMS messages as either spam or non-spam. This project was an incredible journey, leveraging various machine learning techniques to achieve high accuracy in text classification.

🔍 Project Overview:
The objective was to create a robust text classification model to effectively identify and classify SMS messages. By experimenting with different machine learning models, I aimed to find the most efficient approach for this task.

📊 Dataset Description:
Dataset: SMS messages labeled as "spam" or "ham" (non-spam).
Features:
Message:The text content of the SMS.
Label: Classification label ("spam" or "ham").

🛠️ Data Preprocessing:
1. Text Cleaning:
   - Removed punctuation, special characters, and numbers.
   - Converted text to lowercase.
2. Tokenization:
   - Split messages into tokens (words).
3. Stop Word Removal:
   - Removed common, non-informative words.
4. Stemming/Lemmatization:
   - Reduced words to their root form.
5. Feature Extraction:
   - Bag of Words (BoW): Converted text into token counts.
   - TF-IDF: Weighted tokens based on their importance.

🤖 Model Selection:
Evaluated several models:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)
4. Gradient Boosting
5. Naive Bayes
6. K-Nearest Neighbors (KNN)

📈 Model Training and Evaluation:
1. Splitting Data:
   - Training set: 80%
   - Testing set: 20%
2. Evaluation Metrics:
   - Accuracy: Proportion of correctly classified messages.
   - Precision: Proportion of true positives out of predicted positives.
   - Recall: Proportion of true positives out of actual positives.
   - F1 Score: Harmonic mean of precision and recall.
3. Performance Results:
| Model               | Accuracy | Precision | Recall  | F1 Score |
|---------------------|----------|-----------|---------|----------|
| Logistic Regression | 0.9704   | 0.9685    | 0.9982  | 0.9831   |
| Random Forest       | 0.9500   | 0.9457    | 0.9995  | 0.9719   |
| SVM                 | 0.9708   | 0.9688    | 0.9984  | 0.9834   |
| Gradient Boosting   | 0.9538   | 0.9530    | 0.9956  | 0.9738   |
| Naive Bayes         | 0.9805   | 0.9813    | 0.9964  | 0.9888   |
| KNN                 | 0.8712   | 0.8703    | 1.0000  | 0.9307   |

🔍 Analysis:
- Naive Bayes: Best performance with highest accuracy (0.9805), precision (0.9813), recall (0.9964), and F1 score (0.9888).
- SVM: High accuracy (0.9708) and F1 score (0.9834).
- Logistic Regression: Solid performance with accuracy (0.9704) and F1 score (0.9831).
- Random Forest: Moderate accuracy (0.9500) but very high recall (0.9995).
- Gradient Boosting: Good performance with accuracy (0.9538) and F1 score (0.9738).
- KNN: Lowest accuracy (0.8712) but perfect recall (1.0000).

🔧 Best Parameters:
For Logistic Regression:
- Best Parameters: {'C': 10, 'max_iter': 100}
- Best Score: 0.9722

💡 Recommendations:
- Best Overall Model: Naive Bayes for its superior performance.
- Alternative High Performers: SVM and Logistic Regression for their balanced precision and recall.
- Considerations: KNN has perfect recall but lower accuracy and precision, making it less ideal.

🏁 Conclusion:
Naive Bayes emerged as the most effective model for classifying SMS messages as spam or non-spam, thanks to its outstanding accuracy, precision, recall, and F1 score. SVM and Logistic Regression also showed strong performance and can be considered valuable alternatives. Future work could explore more advanced models or techniques to further enhance performance and deploy the model for practical use.

🔗 References:
- Dataset: mail_data
- Libraries Used: pandas, numpy, scikit-learn, nltk, re

