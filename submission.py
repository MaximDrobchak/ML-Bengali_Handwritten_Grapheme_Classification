import pandas as pd
from tensorflow.keras.models import load_model

from load_preprocessing_data import get_load_test_dataset
from constants import TEST_FILENAMES

model = load_model('./data/save_model/model_20.h5')
model.load_weights('./data/save_model/model_weights.20.hdf5')

test = get_load_test_dataset(TEST_FILENAMES)
test = test.batch(1)

predicts = []
for i , (image, label) in enumerate(test):
    predict =  model.predict(image)
    preds = []
    for pred in predict:
        preds += [np.argmax(pred, axis=1).tolist()[0]]
    predicts += [[preds, label[0].numpy().decode("utf-8")]]
predicts

row_ids = []
target = []
for pred in predicts:
    row_id = pred[1].split('.')[0]
    consonant = row_id+'_consonant_diacritic'
    root = row_id+'_grapheme_root'
    vowel = row_id+'_vowel_diacritic'
    row_ids.append(consonant)
    target.append(pred[0][2])
    row_ids.append(root)
    target.append(pred[0][0])
    row_ids.append(vowel)
    target.append(pred[0][1])

df_sample = pd.DataFrame({
    'row_id': row_ids,
    'target':target
},columns=['row_id','target'])

df_sample.to_csv('submission.csv',index=False)
df_sample