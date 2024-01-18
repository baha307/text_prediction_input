import numpy as np
import tensorflow as tf
from google.colab import files

# Upload a file using the file picker
uploaded = files.upload()

# Read the content of the uploaded file
file_name = next(iter(uploaded))
file_content = uploaded[file_name].decode('utf-8')

# Split the content into sentences
sentences = file_content.split('\n')

# Create a tokenizer to convert words to numerical indices
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)

# Convert text to sequences of numerical indices
sequences = tokenizer.texts_to_sequences(sentences)

# Create input sequences and target sequences
input_sequences = []
target_sequences = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i])
        target_sequences.append(sequence[i])

# Pad sequences for consistent input size
max_len = max(len(seq) for seq in input_sequences)
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Convert to numpy arrays
x = np.array(input_sequences)
y = np.array(target_sequences)

# Create a more complex LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_len),
    tf.keras.layers.LSTM(150, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Compile the model with an increased learning rate
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

# Train the model for more epochs
model.fit(x, y, epochs=50, verbose=1)

# Predict next word function using the trained model
def predict_next_word(input_text: str) -> str:
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences([input_sequence], maxlen=max_len, padding='pre')
    predicted_index = np.argmax(model.predict(input_sequence), axis=-1)[0]
    predicted_word = tokenizer.index_word[predicted_index]
    return predicted_word

# Predictive text input loop
while True:
    user_input = input('> ')
    if user_input.lower() == 'exit':
        break
    predicted_word = predict_next_word(user_input)
    print(user_input + ' ' + predicted_word)
