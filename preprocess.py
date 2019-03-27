def load_clean_captions(filename, dataset):
	
	doc = read_file(filename)
	captions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		
		image_id, image_desc = tokens[0], tokens[1:]
		
		if image_id in dataset:
			# create list
			if image_id not in captions:
				captions[image_id] = list()
			
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			
			captions[image_id].append(desc)
	return captions

train_captions = load_clean_captions('captions.txt', train)
print('Descriptions: train=%d' % len(train_captions))

def preprocess(image_path):
    
    img = image.load_img(image_path, target_size=(299, 299))
    
    x = image.img_to_array(img)
    
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    return x

model = InceptionV3(weights='imagenet')

model_new = Model(model.input, model.layers[-2].output)

def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) 
    return fea_vec