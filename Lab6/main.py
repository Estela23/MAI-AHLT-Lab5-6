from learner import learn
from classifier import predict

train_dir = "../data/train/"
validation_dir = "../data/devel/"
test_dir = "../data/test/"

model_name = "first_try_DDI"    # Update each time the name of the model to a more explicative name

outfile = "Lab6/results/output-first_try_DDI.txt"

# Parse data in the xml files and train a model
learn(train_dir, validation_dir, model_name)

# Predict with the trained model about the data in test_dir
predict(model_name, test_dir, outfile)
