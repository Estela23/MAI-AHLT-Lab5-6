def learn ( traindir , validationdir , modelname ) :
'''
learns a NN model using traindir as training data , and validationdir
as validation data . Saves learnt model in a file named modelname
'''
# load train and validation data in a suitable form
traindata = load_data ( traindir )
valdata = load_data ( validationdir )

# create indexes from training data
max_len = 100
idx = create_indexs ( traindata , max_len )

# build network
model = build_network (idx)

# encode datasets
Xtrain = encode_words ( traindata , idx )
Ytrain = encode_labels ( traindata , idx )
Xval = encode_words ( valdata , idx )
Yval = encode_labels ( valdata , idx )

# train model
model.fit( Xtrain , Ytrain , validation_data =( Xval , Yval ))

# save model and indexs , for later use in prediction
save_model_and_indexs (model , idx , modelname )
