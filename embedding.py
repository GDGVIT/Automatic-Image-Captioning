import pickle

start = time()
encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)
print("Time taken in seconds =", time()-start)

# Save the train features to disk
#"/content/gdrive/My Drive/flickr_dataset/Flickr8k_text/Flickr8k.token.txt"
with open("/content/gdrive/My Drive/flickr_dataset/pickle/encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)

start = time()
encoding_test = {}
for img in test_img:
    encoding_test[img[len(images):]] = encode(img)
print("Time taken in seconds =", time()-start)
print(len(encoding_test))

# Save the test features to disk
with open("/content/gdrive/My Drive/flickr_dataset/pickle/encoded_test_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)

train_features = load(open("/content/gdrive/My Drive/flickr_dataset/pickle/encoded_train_images.pkl", "rb"))
print('Photos: train=%d' % len(train_features))

test_features = load(open("/content/gdrive/My Drive/flickr_dataset/pickle/encoded_test_images.pkl", "rb"))
print('Photos: train=%d' % len(train_features))

all_train_captions = []
for key, val in train_captions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1 
vocab_size


def to_lines(captions):
	all_desc = list()
	for key in captions.keys():
		[all_desc.append(d) for d in captions[key]]
	return all_desc

# calculate the length of the description with the most words
def max_length(captions):
	lines = to_lines(captions)
	return max(len(d.split()) for d in lines)

# determine the maximum sequence length
max_length = max_length(train_captions)
print('Description Length: %d' % max_length)

def data_generator(captions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
   
    while 1:
        for key, desc_list in captions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list: 
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
               
                for i in range(1, len(seq)):
                   
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0

glove_dir = '/content/gdrive/My Drive/flickr_dataset/glove.6B'
embeddings_index = {} # empty dictionary
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    #if i is less than maximum words
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
       
        embedding_matrix[i] = embedding_vector

embedding_matrix.shape