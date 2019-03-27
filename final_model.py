# defining the model

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.summary()

model.layers[2]

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

epochs = 10
number_pics_per_bath = 3
steps = len(train_captions)//number_pics_per_bath

for i in range(epochs):
    generator = data_generator(train_captions, train_features, wordtoix, max_length, number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

model.save()

for i in range(epochs):
    generator = data_generator(train_captions, train_features, wordtoix, max_length, number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('./model_weights/model_' + str(i) + '.h5')

model.optimizer.lr = 0.0005
epochs = 10
number_pics_per_bath = 6
steps = len(train_captions)//number_pics_per_bath

for i in range(epochs):
    generator = data_generator(train_captions, train_features, wordtoix, max_length, number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('./model_weights/model_' + str(i) + '.h5')

model.save_weights("/content/gdrive/My Drive/flickr_dataset/model_weights/model_30.h5")

model.load_weights("/content/gdrive/My Drive/flickr_dataset/model_weights/model_30.h5")

images = "/content/gdrive/My Drive/flickr_dataset/Flickr8k_Dataset/Flicker8k_Dataset/"