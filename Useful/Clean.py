import Clean_data
import pandas as pd

train_data = pd.read_csv("./Data/test.csv")
train_data.fillna('NO QUESTION', inplace = True)
#train_small = train_data[0:10000]
#train_small['question1'] = train_small['question1'].apply(lambda text: clean_class.clean(text))
#train_small['question2'] = train_small['question2'].apply(lambda text: clean_class.clean(text))
#train_small.to_csv("Cleaned train data small.csv", index=False)
clean_class = Clean_data.Clean_data()
train_data['question1'] = train_data['question1'].apply(lambda text: clean_class.clean(text))
train_data['question2'] = train_data['question2'].apply(lambda text: clean_class.clean(text))
train_data.to_csv("Cleaned_test_complete.csv", index=False)
