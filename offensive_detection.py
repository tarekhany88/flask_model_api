import re
import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.models import load_model
nltk.download('punkt')
nltk.download('wordnet')


ar_stopwords = '''
أنفسنا مثل حيث ذلك بشكل لدى ألا عن إلي ب لنا وقالت فقط الذي الذى ا هذا غير أكثر اي أنا أنت ايضا اذا كيف وكل أو اكثر أي أن منه وكان وفي تلك إن سوف حين نفسها هكذا قبل حول منذ هنا عندما على ضمن لكن فيه عليه قليل صباحا لهم بان يكون بأن أما هناك مع فوق بسبب ما لا هذه و فيها ف ولم ل آخر ثانية انه من الان جدا به بن بعض حاليا بها هم أ كانت هي لها نحن تم أنفسهم ينبغي إلى فان وقد تحت' عند وجود الى فأن الي او قد خارج إنه اى مرة هؤلاء أنها إذا فهي فهى كل يمكن جميع أنفسكم فعل كان ثم لي الآن وقال فى في ديك لم لن له تكون الذين ليس التى التي أنه وان بعد حتى ان دون وأن لماذا يجري كلا إنها لك ضد وإن فهو انها منها أى لديه ولا بين خلال وما اما عليها بعيدا كما نفسي نحو هو نفسك نفسه انت ولن إضافي لقاء وكانت هى فما أيضا إلا معظم ومن إما الا بينما وهي وهو وهى
'''

ar_stopwords=nltk.word_tokenize(ar_stopwords)
# print("length of stopwords is: ",len(ar_stopwords))
tokenizer = Tokenizer()
model = load_model("model.h5")

def process_text(text):
    stemmer = nltk.ISRIStemmer()
    word_list = nltk.word_tokenize(text)
    # remove arabic stopwords
    word_list = [w for w in word_list if not w in ar_stopwords]
    # remove digits
    word_list = [w for w in word_list if not w.isdigit()]
    # stemming
    word_list = [stemmer.stem(w) for w in word_list]
    return ' '.join(word_list)


def clean_text(text):
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى",
              "\\", '\n', '\t', '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا",
               "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ', ' ! ']
    # remove tashkeel
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, "", text)

    longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(longation, subst, text)

    text = re.sub(r"[^\w\s]", '', text)
    # remove english words
    text = re.sub(r"[a-zA-Z]", '', text)
    # remove spaces
    text = re.sub(r"\d+", ' ', text)
    text = re.sub(r"\n+", ' ', text)
    text = re.sub(r"\t+", ' ', text)
    text = re.sub(r"\r+", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    # remove repetetions
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    text = text.strip()

    return process_text(text)

def predict_sentiment(input):
  input=clean_text(input)
  model_input=tokenizer.texts_to_sequences(input)
  model_input=pad_sequences(model_input, padding='post', maxlen=300)
  pred = model.predict(model_input, verbose=True)
  pred=pred.reshape(pred.shape[0],)[0]
  return float(pred)


def feedback_learing(text,correct_pred):
  X=clean_text(text)
  tokenizer.fit_on_texts(X)
  X = tokenizer.texts_to_sequences(X)
  X = pad_sequences(X, padding='post', maxlen=300)
  Y = np.asarray([correct_pred] * len(X)).astype('float32').reshape((-1, 1))
  Z= model.train_on_batch(X, Y)
  print(Z)
  # model.save('model.h5')