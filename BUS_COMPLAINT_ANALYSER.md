# Bus Complaint Analyser — Academic Notes

## Objective

Automatically route a passenger complaint to one of six operational departments.

## NLP pipeline

1. Normalize case, URLs, email addresses, punctuation, and whitespace.
2. Tokenize the text.
3. Build TF-IDF word n-gram features.
4. Build TF-IDF character n-gram features.
5. Combine both feature spaces.
6. Train Logistic Regression with balanced class weights.
7. Return probability ranking rather than only a class label.
8. Flag low-confidence cases for human review.
9. Retrieve similar labeled complaints using cosine similarity.
10. Add transparent lexicon-based sentiment and keyword evidence.

## Why word + character TF-IDF?

Word n-grams capture phrases such as `charged twice`, `bus arrived`, and `driver phone`.
Character n-grams make the model less brittle to small spelling variations and morphology.

## Limitations

The included dataset is intentionally tiny: 48 examples, evenly distributed across six classes.
The system is therefore an academic demonstration. Confidence values should not be interpreted as calibrated probabilities or guaranteed correctness.

## Future research

- Expand the labeled corpus to thousands of complaints.
- Create an annotation guide and measure inter-annotator agreement.
- Add multilingual support for Marathi/Hindi/English.
- Compare Logistic Regression, Linear SVM, and transformer embeddings.
- Calibrate probabilities.
- Add time-based evaluation and drift monitoring.
- Store predictions and human corrections for continuous improvement.
