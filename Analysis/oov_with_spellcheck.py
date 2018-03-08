from autocorrect import spell
import pandas as pd
import timeit

start_time = timeit.default_timer()
df_with_OOV = pd.read_csv('oov_words_per_question.csv')

def convert_to_list(oov_string):
    oov_string = oov_string.split("'")
    oov_string = oov_string[1::2]
    for i in range(len(oov_string)):
        print oov_string[i] + ' ',
        oov_string[i] = spell(oov_string[i])
        print oov_string[i]
    return oov_string

df_with_OOV['out_of_vocab'] = df_with_OOV['out_of_vocab'].apply(lambda oov_string: convert_to_list(oov_string))
#df_with_OOV['out_of_vocab'] = df_with_OOV['out_of_vocab'].apply(lambda string: [spell(word) for word in string])

elapsed = timeit.default_timer() - start_time
print(str(elapsed))

start_time = timeit.default_timer()
df_with_OOV.to_csv('OOV_after_spellcheck.csv', index=False)
elapsed = timeit.default_timer() - start_time
print(str(elapsed))
