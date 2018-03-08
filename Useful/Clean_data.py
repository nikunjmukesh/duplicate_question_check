import re

class Clean_data():
    stop = ['it', 'is', 'the', 'had', 'have', 'has', 'i', 'a', 'and', 'our', 'are',
        'you', 'do', 'my', 'am', 'were', 'was', 'by', 'until', 'but', 'my',
        'myself', 'itself', 'them', 'themself', 'themselves', 'at', 'ours',
        'do', 'his', 'ourself', 'ourselves', 'must', 'we', 'be', 'here', 'there',
        'some', 'for', 'while', 'should', 'her', 'hers', 'their', 'theirs', 'by',
        'on', 'about', 'could', 'would', 'of', 'against', 'more', 'him', 'that',
        'with', 'than', 'those', 'he', 'me', 'in', 'any', 'if', 'again', 'no',
        'same', 'other', 'such', 'a', 'yours', 'your', 'so', 'having', 'once']  
    punctuation = ['?', ',']
    punct = ["'"]
    def clean(self, text):
        text = text.lower()
        text = re.sub(r"(\d+)k\s", r"\g<1>000", text)
        #text = re.sub(r"(\d+)K", r"\g<1>000", text)
        text = re.sub(r"(\d+)\sk\s", r"\g<1>000", text)
        #text = re.sub(r"(\d+)\sK", r"\g<1>000", text)
        text = re.sub(r"(\d+\s)-(\s\d+)", r"\1to\2 ", text)
        text = re.sub(r"(\d+)-(\d+)", r"\1 to \2", text)
        text = re.sub(r"\.\.\.*", ".", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" u.k. ", " england ", text)
        text = re.sub(r" u.k ", " england ", text)
        text = re.sub(r" u.s ", " america ", text)
        text = re.sub(r" u.s. ", " america ", text)
        text = re.sub(r" u.s.a", " america", text)
        text = re.sub(r" u.s.a.", " america", text)
        text = re.sub(r" m.tech ", "mtech", text)
        text = re.sub(r" b.tech ", "btech", text)
        text = re.sub(r" m.tech. ", "mtech", text)
        text = re.sub(r" b.tech. ", "btech", text)
        text = ''.join([i for i in text if i not in self.punctuation])
        text = " ".join(text.split())
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"\'s", " is ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r" ive ", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r" e g ", " example ", text)
        text = re.sub(r" e.g ", " example ", text)
        text = re.sub(r" e.g. ", " example ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e-mail", "email", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"quikly", "quickly", text)
        text = re.sub(r"imrovement", "improvement", text)
        text = re.sub(r"intially", "initially", text)
        text = re.sub(r" dms ", "direct messages ", text)  
        text = re.sub(r"demonitization", "demonetization", text) 
        text = re.sub(r"actived", "active", text)
        text = re.sub(r"kms", " kilometers ", text)
        text = re.sub(r" upvotes ", " up votes ", text)
        text = re.sub(r" iPhone ", " phone ", text)
        text = re.sub(r"\0rs ", " rs ", text) 
        text = re.sub(r"calender", "calendar", text)
        text = re.sub(r"ios", "operating system", text)
        text = re.sub(r"programing", "programming", text)
        text = re.sub(r"bestfriend", "best friend", text)
        text = re.sub(r"III", "3", text) 
        text = re.sub(r"the us", "america", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" j.k. ", " jk ", text)
        text = re.sub(r" j.k ", " jk ", text)
        text = re.sub(r"\scse\s", " computer science engineering ", text)
        text = re.sub(r"\scs\s", " computer science ", text)
        text = re.sub(r"(\d+)k", r"\g<1>000", text)
        text = re.sub(r" rs(\d)", r" rupees \1", text)
        text = re.sub(r" rs.(\d)", r" rupees \1", text)
        text = re.sub(r" rs. (\d)", r" rupees \1", text)
        text = re.sub(r"\srs\s", " rupees ", text)
        text = re.sub(r"(\d)rs", r"\1 rupees", text)
        text = re.sub(r"inr(\d)", r"rupees \1", text)
        text = re.sub(r"inr (\d)", r"rupees \1", text)
        text = re.sub(r"(\d)inr", r"\1 rupees", text)
        text = re.sub(r"(\d) inr", r"\1 rupees", text)
        if len(text)>=1:
            if (text[-1] == "." ):
                text_list = list(text)
                text_list.pop()
                text = "".join(text_list)
        text = re.sub(r"(\S)\.\s", r"\1 . ", text)
        text = re.sub(r"(\S)\.\n", r"\1 . ", text)
        text = re.sub(r"(\S)\.\0", r"\1 . ", text)
        text = re.sub(r"\s\.\s", " ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" uk ", " england ", text)
        text = ''.join([i for i in text if i not in self.punct])
        #to remove stop words
        text = text.split()
        text = [word for word in text if not word in self.stop]
        text = " ".join(text)
        return text
    
    
    
'''train_data = pd.read_csv("../../Data/train.csv")
train_small = train_data[0:10000]
clean_class = Clean_data()
train_small['question1'] = train_small['question1'].apply(lambda text: clean_class.clean(text))
train_small['question2'] = train_small['question2'].apply(lambda text: clean_class.clean(text))
train_small.to_csv("Cleaned train data small.csv", index=False)'''
    
    
    
    
    
    
        
