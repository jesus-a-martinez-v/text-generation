import os
import random
import re
import sys
from io import BytesIO

import PIL
import keras.backend as K
import numpy as np
import scipy.misc
from IPython.display import Image, display
from PIL import ImageDraw
from PIL.Image import Image
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Input, Model
from keras.optimizers import RMSprop

SEQUENCE_SIZE = 160


def char_rnn_model(num_chars, num_layers, num_nodes=512, dropout=0.1):
    input_layer = Input(shape=(None, num_chars), name='input')

    previous_layer = input_layer
    for layer_index in range(num_layers):
        lstm_layer = LSTM(num_nodes, return_sequences=True, name=f'lstm_layer_{layer_index + 1}')(previous_layer)

        if dropout:
            previous_layer = Dropout(dropout)(lstm_layer)
        else:
            previous_layer = lstm_layer

    output_layer = TimeDistributed(Dense(num_chars, name='dense', activation='softmax'))(previous_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    optimizer = RMSprop(lr=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def data_generator(text_corpus, char_to_index, batch_size, sequence_size):
    number_of_chars = len(char_to_index)
    
    # Batch examples and labels
    X = np.zeros(shape=(batch_size, sequence_size, number_of_chars))
    y = np.zeros(shape=(batch_size, sequence_size, number_of_chars))

    while True:
        for row in range(batch_size):
            index = random.randrange(len(text_corpus) - sequence_size - 1)
            chunk = np.zeros(shape=(sequence_size + 1, number_of_chars))

            for sequence_index in range(sequence_size + 1):
                character_in_text = text_corpus[index + sequence_index]
                character_index = char_to_index[character_in_text]
                chunk[sequence_index, character_index] = 1  # This is one hot encoding the vector corresponding to this seq

            # Each letter in the sequence is the label for the previous one.
            X[row, :, :] = chunk[:sequence_size]
            y[row, :, :] = chunk[1:]

        yield X, y


def find_python(root_directory):
    matches = []
    for root, directory_names, file_names in os.walk(root_directory):
        for file_name in file_names:
            if file_name.endswith('.py'):  # It is a Python source file.
                matches.append(os.path.join(root, file_name))

    return matches


root_directory = random.__file__.rsplit('/', 1)[0]  # Many modules are stored in the same directory as the random module.
sources = find_python(root_directory)
print(len(sources))


def replacer(value):
    value = ''.join(ch for ch in value if ord(ch) < 127)

    if ' ' not in value:
        return value

    if sum(1 for ch in value if ch.isalpha()) > 6:
        return 'MSG'

    return value


def replace_literals(st):
    res = []
    start_text = 0
    start_quote = 0
    i = 0
    quote = ''

    while i < len(st):
        if quote:
            if st[i: i + len(quote)] == quote:
                quote = ''
                start_text = i
                res.append(replacer(st[start_quote:i]))
        elif st[i] in '"\'':
            quote = st[i]

            if i < len(st) - 2 and st[i + 1] == st[i + 2] == quote:
                quote = 3 * quote

            start_quote = i + len(quote)
            res.append(st[start_text:start_quote])

        if st[i] == '\n' and len(quote) == 1:
            start_text = i
            res.append(quote)
            quote = ''

        if st[i] == '\\':
            i += 1

        i += 1

    return ''.join(res) + st[start_text:]


replace_literals('this = "wrong\n')

COMMENT_RE = re.compile("#.*")
python_code = []

for file_name in sources:
    try:
        with open(file_name, 'r') as f:
            source = f.read()
    except UnicodeDecodeError:
        print(f'Could not read {file_name}')

    source = replace_literals(source)
    source = COMMENT_RE.sub('', source)
    python_code.append(source)

python_code = '\n\n\n'.join(python_code)
len(python_code)

python_chars = list(sorted(set(python_code)))
python_char_to_index = {ch: index for index, ch in enumerate(python_chars)}
print(len(python_chars))

python_model = char_rnn_model(len(python_chars), num_layers=2, num_nodes=640, dropout=0)
print(python_model.summary())

early = EarlyStopping(monitor='loss', min_delta=0.03, patience=3, mode='auto')

BATCH_SIZE = 256
generator = data_generator(python_code, python_char_to_index, batch_size=BATCH_SIZE, sequence_size=SEQUENCE_SIZE)
python_model.fit_generator(
    generator,
    epochs=60,
    callbacks=[early],
    steps_per_epoch=2 * len(python_code) / (BATCH_SIZE * 160)
)


def generate_code(model, start_with='\ndef ', end_with='\n\n', diversity=1.0):
    generated = start_with

    yield generated

    for i in range(2000):
        x = np.zeros(shape=(1, len(generated), len(python_chars)))
        for t, char in enumerate(generated):
            x[0, t, python_char_to_index[char]] = 1.0
        predictions = model.predict(x)[0]

        predictions = np.asarray(predictions[len(generated) - 1]).astype('float64')
        predictions = np.log(predictions) / diversity
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probabilities = np.random.multinomial(1, predictions, 1)
        next_index = np.argmax(probabilities)
        next_char = python_chars[next_index]

        yield next_char

        generated += next_char

        if generated.endswith(end_with):
            break


for i in range(20):
    for ch in generate_code(python_model):
        sys.stdout.write(ch)
    print()

BATCH_SIZE = 512

flat_model = char_rnn_model(len(python_chars), num_layers=1, num_nodes=512, dropout=0)

early = EarlyStopping(monitor='loss', min_delta=0.03, patience=3, mode='auto')

generator = data_generator(python_code, python_char_to_index, batch_size=BATCH_SIZE, sequence_size=SEQUENCE_SIZE)
flat_model.fit_generator(
    generator,
    epochs=60,
    callbacks=[early],
    steps_per_epoch=2 * len(python_code) / (BATCH_SIZE * 160)
)

example_code = 'if a == 2:\n    b = 1\nelse:\n    b = 2\n'


def activations(model, code):
    x = np.zeros((1, len(code), len(python_char_to_index)))
    for t, char in enumerate(code):
        x[0, t, python_char_to_index[char]] = 1.0

    output = model.get_layer('lstm_layer_1').output

    f = K.function([model.input], [output])
    return f([x])[0][0]


act = activations(flat_model, example_code)
print(act.shape)


def interesting_neurons(act):
    result = []

    for n in np.argmax(act, axis=1):
        if n not in result:
            result.append(n)

    return result


neurons = interesting_neurons(act)
print(len(neurons))


def visualize_neurons(neurons, code, act, cell_size=12):
    image = np.full(shape=(len(neurons) + 1, len(code), 3), fill_value=128)
    scores = (act[:, neurons].T + 1) / 2
    image[1:, :, 0] = 255 * (1 - scores)
    image[1:, :, 1] = 255 * scores

    f = BytesIO()
    image = scipy.misc.imresize(image, float(cell_size), interp='nearest')
    pil_img = PIL.Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    for index, ch in enumerate(code):
        draw.text((index * cell_size + 2, 0), ch)

    pil_img.save(f, 'png')
    return Image(data=f.getvalue())


img = visualize_neurons(neurons, example_code, act)
display(img)


def image_for_code(code):
    act = activations(flat_model, code)
    neurons = interesting_neurons(act)
    return visualize_neurons(neurons, code, act)


display(image_for_code('if (a == 2) and ((b == 1) or (c == 2)):'))

code = 'if (a == 2) and ((b == 1) or (c == 2)):'
mask = '   ________     ____________________ '
act = activations(flat_model, code)
positive = [index for index, ch in enumerate(mask) if ch == '_']
negative = [index for index, ch in enumerate(mask) if ch != '_']

neurons = np.argsort(act[positive].sum(axis=0) - act[negative].sum(axis=0))[-5:]

img = visualize_neurons(neurons, code, act)
display(img)
