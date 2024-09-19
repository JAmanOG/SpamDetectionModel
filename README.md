# Spam Detection Model

This project aims to build a machine learning model to classify emails as spam or non-spam (ham) using TensorFlow and Keras. The model is trained on a dataset of labeled emails and tested on new email samples.

## Overview

- **Dataset**: Small example dataset of emails.
- **Model**: LSTM-based neural network.
- **Preprocessing**: Includes text cleaning, tokenization, and padding.
- **Training**: Model is trained with an 80-20 split for training and validation.

## Prerequisites

Ensure you have the following libraries installed:

```bash
pip install tensorflow numpy scikit-learn
```

## Data Preparation

The dataset used in this project consists of labeled emails:

- **Spam**: Emails that are typically unsolicited or promotional.
- **Non-Spam (Ham)**: Regular emails such as invitations or important messages.

The emails and labels are as follows:

```python
emails = [
    "Buy cheap watches! Free shipping!",
    "Meeting for lunch today?",
    "Claim your prize! You've won $1,000,000!",
    "Important meeting at 3 pm.",
    "You're invited to a dinner party at my place.",
    "Exclusive deal just for you!",
    "How about a catch-up call this weekend?",
    "Congratulations! You've won a prize!"
]
labels = [1, 0, 1, 0, 0, 1, 0, 1]  # 1 for spam, 0 for non-spam
```

## Text Preprocessing

1. **Lowercasing**: Converts all text to lowercase.
2. **Remove Numbers**: Eliminates numerical digits.
3. **Remove Punctuation**: Strips out punctuation marks.

```python
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text
```

## Model Architecture

The model used is an LSTM-based neural network:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Training

The model is trained using:

```python
model.fit(np.array(X_train), y_train, epochs=10, validation_data=(np.array(X_test), y_test), verbose=1)
```

### Prediction

After training, you can test the model with new email samples:

```python
sample_file_path = "NSpam.txt"
with open(sample_file_path, "r", encoding="utf-8") as file:
    sample_email_text = file.read()

sample_email_text = preprocess_text(sample_email_text)
sample_sequences = tokenizer.texts_to_sequences([sample_email_text])
sample_email_padded = pad_sequences(sample_sequences, maxlen=max_len, padding="post", truncating="post")

prediction = model.predict(sample_email_padded)
threshold = 0.5

if prediction[0][0] > threshold:
    print(f"Sample email ({sample_file_path}): SPAM")
else:
    print(f"Sample email ({sample_file_path}): NOT SPAM")
```

## Troubleshooting

- **Model Accuracy**: Ensure your dataset is balanced and sufficiently large. Fine-tune hyperparameters and model architecture if needed.
- **File Errors**: Make sure the file paths are correct and files are accessible.

## Future Improvements

- Expand the dataset with more diverse examples.
- Implement more advanced models like BERT for better accuracy.
- Explore additional preprocessing steps and hyperparameter tuning.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This updated `README.md` file includes:
- **Enhanced preprocessing** details.
- **Model architecture** and training instructions.
- **Example of testing** and handling file errors.
- **Suggestions for future improvements**.

Feel free to adjust any sections based on your specific needs or additional features.
