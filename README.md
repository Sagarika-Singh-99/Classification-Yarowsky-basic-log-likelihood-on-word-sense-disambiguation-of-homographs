# Classification-Yarowsky-basic-log-likelihood-on-word-sense-disambiguation-of-homographs

## Implementing Yarowsky’s basic log-likelihood decision list classifier on word sense dis- ambiguation of homographs. The goal is to classify the features (Bass and Sake) to their correct sense (fish or music, beer or cause).
Bass can mean either fish or music.
Sake can mean either beer or cause.

## Explanation of what we did conceptually and any implementation decisions we took:
1. Prepared the text by converting them into lowercase, removed punctuation’s and tokenized it.
2. Extracted feature from the context around the target word.
(a) 5 words before an after target word
(b) words at specific positions around the target word (-1, +1, -2, and +2)
(c) Bi-grams and Tri-grams were also taken into consideration. This decision was taken later on as our initial accuracy was significantly low compared to the baseline accuracy.
3. Log-likelihood was computed then. We made the decision to apply Laplace Smoothing for unseen features ( 1 was added to the numerator and 2 to the denominator).
4. Ranked decision list was calculated then. It was built by ranking features according to their log-likelihood scores.
5. Classification of test data based on the ranked dl. We used ”:” to split the data between feature (bass/sake) and context.
6. Baseline model is generated to serve as a benchmark
7. Multiple evaluation metrics were calculated.
