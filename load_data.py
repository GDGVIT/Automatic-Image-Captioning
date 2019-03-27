# summarize vocabulary
vocabulary = to_vocabulary(captions)
print('Original Vocabulary Size: %d' % len(vocabulary))

def save_captions(captions, filename):
	lines = list()
	for key, desc_list in captions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

save_captions(captions, 'captions.txt')

def load_set(filename):
	doc = read_file(filename)
	dataset = list()
	
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load training dataset
filename = ".../Flickr_8k.trainImages.txt"
train = load_set(filename)
print('Dataset: %d' % len(train))



images = "/content/gdrive/My Drive/flickr_dataset/Flicker8k_Dataset/"
# Create a list of all image names in the directory
img = glob.glob(images + '*.jpg')
print(img)


train_images_file = "/content/gdrive/My Drive/flickr_dataset/Flickr8k_text/Flickr_8k.trainImages.txt"
# Read the train image names in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))


train_img = []

for i in img: #
    if i[len(images):] in train_images: # Checking if the image is in training set
        train_img.append(i)

test_images_file = '/content/gdrive/My Drive/flickr_dataset/Flickr8k_text/Flickr_8k.testImages.txt'

test_images = set(open(test_images_file, 'r').read().strip().split('\n'))


test_img = []

for i in img: 
    if i[len(images):] in test_images:
        test_img.append(i)