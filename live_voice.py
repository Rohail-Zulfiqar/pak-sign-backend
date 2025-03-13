
import speech_recognition as sr
import pyaudio
import wave
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import stanza
import os


app = Flask(__name__)
CORS(app)

def record_audio(filename, record_seconds=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    #recording start
    stream = audio.open(format=FORMAT, channels=CHANNELS,rate=RATE, input=True,frames_per_buffer=CHUNK)
    print("Recording started...")

    frames = []

    for i in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording Finished")

    #stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    #save recorded audio
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def speech_to_text(filename):
    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)

    try:
        #eecognizing speech using google Web Speech API
        text = recognizer.recognize_google(audio, language="ur-PK")
        print("Urdu Text: " + text)
        return text
    except sr.UnknownValueError:
        error_message = "Google API could not understand the audio."
        print(error_message)
        return error_message
        
    except sr.RequestError as e:
        error_message = "Could not request results from Google API. "
        print(error_message)
        return error_message


# API Endpoint to start recording when requested by the frontend
@app.route('/start-recording', methods=['POST'])
def start_recording():
    try:
        audio_file = "recorded_audio.wav"
        # duration = request.json.get("duration", 5)  # Default to 5 seconds if not specified
        record_audio(audio_file, record_seconds=6)
        return jsonify({"message": "Recording completed", "file": audio_file})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API Endpoint to process the recorded audio (transcribe + translate)
@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        audio_file = "recorded_audio.wav"
        if not os.path.exists(audio_file):
            return jsonify({"error": "Audio file not found"}), 404
        
        urdu_text = speech_to_text(audio_file)
        print(urdu_text)

        # Remove the audio file after processing
        os.remove(audio_file)
        return jsonify({ "urdu_text": urdu_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


glosses = {"تین": "three",
    "ماہ": "month",
    "بعد": "after",
    "دوبارہ": "again",
    "آیئے": "come",
    "اچھا": "good",
    "آنکھ": "eye",
    "سن": "year",
    "ہونا": "to be",
    "کیا": "what",
    "تم": "you (informal)",
    "بہرے": "deaf",
    # "ہو": "become",
    "بھوکے": "hungry",
    "آپ": "you (formal)",
    "مسلمان": "Muslim",
    "غیر": "non",
    "غیر مسلم": "non-Muslim",
    "تیار": "ready",
    "کسی": "someone",
    "دوسرے": "others",
    "ڈاکٹر": "doctor",
    "ہسپتال": "hospital",
    "میں": "in",
    "زیر": "under",
    "علاج": "treatment",
    "کرائے": "rented",
    "دار": "holder",
    "اپنا": "own",
    "ذاتی": "personal",
    "مکان": "house",
    "مشورہ": "advice",
    "لے": "take",
    "رہے": "are",
    "پہلی": "first",
    "دفعہ": "time",
    "عینک": "glasses",
    "پہن": "wear",
    "السلام": "peace",
    "وعلیکم": "upon you",
    "خبر": "news",
    "ایمبولینس": "ambulance",
    "کال": "call",
    "کریں": "do",
    "اپ": "you",
    "مدد": "help",
    "کر": "do",
    "سکتا": "can",
    "ہوں": "am",
    "حکم": "order",
    "ٹوائلٹ": "toilet",
    "استعمال": "use",
    "اپنے": "your",
    "پیروں": "feet",
    "روزانہ": "daily",
    "معائنہ": "inspection",
    "نظر": "sight",
    "کیمیاوی": "chemical",
    "مواد": "materials",
    "وجہ": "reason",
    "آنکھوں": "eyes",
    "زخم": "wound",
    "دروازہ": "door",
    "بند": "closed",
    "کرو": "do",
    "کھڑکی": "window",
    "ادھر": "here",
    "آو": "come",
    "سرجری": "surgery",
    "کے": "of",
    "کے لیئے": "for",
    "کل": "tomorrow",
    "آئیں": "come",
    "کبھی": "sometime",
    "مجھ": "me",
    "ملنے": "meet",
    "مبارک": "congratulations",
    "نہانے": "bathing",
    "ناخن": "nail",
    "تراش": "trim",
    "لیں": "take",
    "کنارے": "edges",
    "تراشنے": "trimming",
    "گریز": "avoid",
    "اس": "this",
    "پوچھا": "asked",
    "کھا": "eat",
    "پاس": "near",
    "ناکام": "failed",
    "ہوئے": "were",
    "سمجھے": "understood",
    "جانوروں": "animals",
    "اجازت": "permission",
    "دیتے": "give",
    "بالکل": "absolutely",
    "ٹھیک": "okay",
    "محسوس": "feel",
    "کرتے": "do",
    "آپکو": "to you",
    "چکؔر": "dizziness",
    "آرہے": "coming",
    "خارش": "itching",
    "ہوتی": "happens",
    "اسکول": "school",
    "جاتے": "go",
    "آپکے": "your",
    "ایک": "one",
    "پالتوجانور": "pet animal",
    "جانور": "animal",
    "کوئی": "someone",
    "اسپرین": "aspirin",
    "اثاثے": "assets",
    "ذمے": "responsible",
    "قرض": "loan",
    "پاؤں": "feet",
    "جلن": "burning",
    "ذیابیطس": "diabetes",
    "سونا": "gold",
    "چاندی": "silver",
    "زیادہ": "more",
    "تجارتی": "commercial",
    "سامان": "goods",
    "جانتے": "know",
    "یہاں": "here",
    "کمرہ": "room",
    "کہاں": "where",
    "تلاش": "search",
    "ٹی وی": "television",
    "چائے": "tea",
    "دیکھنا": "see",
    "چاہتے": "want",
    "انگریزی": "English",
    "بول": "speak",
    "سکتے": "can",
    "سماعت": "hearing",
    "لیتے": "take",
    "سافٹ": "soft",
    "ڈرنک": "drink",
    "پینا": "drink",
    "دودھ": "milk",
    "چینی": "sugar",
    "فلموں": "movies",
    "جانا": "go",
    "ہاتھ": "hand",
    "خاندان": "family",
    "بات": "talk",
    "نہیں": "not",
    "گھبرائیں": "panic",
    "اپنی": "your",
    "زور": "pressure",
    "مت": "don't",
    "ڈالیں": "put",
    "چھونا": "touch",
    "گھاس": "grass",
    "ننگے": "bare",
    "چلیں": "walk",
    "پیر": "foot",
    "اچھی": "good",
    "طرح": "way",
    "خشک": "dry",
    "دورانِ": "during",
    "مطالعہ": "study",
    "آپکی": "your",
    "کمزور": "weak",
    "عید": "Eid",
    "معاف": "forgive",
    "کیجئے": "do",
    "گا": "will",
    "سرخ": "red",
    "موروثی": "hereditary",
    "طبعی": "physical",
    "معلومات": "information",
    "گرمی": "heat",
    "سردی": "cold",
    "احساس": "feeling",
    "چلے": "walk",
    "جاؤ": "go",
    "دوپہر": "afternoon",
    "بخیر": "good",
    "شام": "evening",
    "صبح": "morning",
    "شب": "night",
    "خدا": "God",
    "حافظ": "protector",
    "نیا": "new",
    "سال": "year",
    "دن": "day",
    "گزاریں": "spend",
    "لیا": "taken",
    "ہیلو": "hello",
    "کیسے": "how",
    "آپ": "you (formal)",
    "وہاں": "there",
    "جا": "go",
    "کیسا": "what kind",
    "کتنی": "how much",
    "دور": "far",
    "دیر": "late",
    "کتنے": "how many",
    "وقت": "time",
    "رہو": "stay",
    "گے": "will",
    "کب": "when",
    "یہ": "this",
    "مسئلہ": "problem",
    "مرض": "disease",
    "مبتلا": "afflicted",
    "چشمہ": "spectacles",
    "بچے": "children kids",
    "بچّے": "kids",
    "رات": "night",
    "بار": "time",
    "پیشاب": "urine",
    "کتنا": "how much",
    "قیمت": "price",
    "مرتبہ": "rank",
    "طالب علم": "student",
    "علم": "knowledge",
    "اشاروں": "gestures",
    "زبان": "language",
    "سیکھ": "learn",
    "بیمار": "sick",
    "مترجم": "translator",
    "ہونے": "to be",
    "مجھے": "me",
    "یقین": "belief",
    "پسند": "like",
    "سمجھا": "understood",
    "اندراج": "registration",
    "کارڈ": "card",
    "لانا": "bring",
    "بھول": "forget",
    "گیا": "gone",
    "میرا": "my",
    "سوال": "question",
    "کچھ": "some",
    "دوا": "medicine",
    "خریدنا": "buy",
    "خریداری": "purchase",
    "کرنے": "to do",
    "جا": "go",
    "نا": "not",
    "کام": "work",
    "جلدی": "quickly",
    "کرنا": "to do",
    "چھٹی": "leave",
    "کیلئے": "for",
    "درخواست": "application",
    "دینی": "to give",
    "بینک": "bank",
    "جانے": "to go",
    "ضرورت": "need",
    "شرٹ": "shirt",
    "دھونے": "wash",
    "تھوڑا": "little",
    "سا": "bit",
    "بولتا": "speaks",
    "آج": "today",
    "کمپیوٹر": "computer",
    "کلاس": "class",
    "شروع": "start",
    "پوری": "full",
    "بہروں": "deaf people",
    "کلب": "club",
    "دورہ": "visit",
    "چاہتا": "wants",
    "بہرا": "deaf",
    "پیدا": "born",
    "ہوا": "happened",
    "تھا": "was",
    "بچوں": "children",
    "سننے": "listening",
    "لئے": "for",
    "جلد": "soon",
    "ہی": "just",
    "واپس": "return",
    "آؤں": "come (I)",
    "سکتی": "can (she)",
    "انٹرویو": "interview",
    "چاہوں": "want (I)",
    "ملتا": "meets",
    "بلڈ": "blood",
    "پریشر": "pressure",
    "کم": "low",
    "وہ": "he/she",
    "زکوۃ": "charity",
    "عطیہ": "donation",
    "مستحق": "deserving",
    "ٹوٹا": "broken",
    "سچ": "truth",
    "ملاقات": "meeting",
    "کیا": "what",
    "ئی": "came",
    "مل": "found",
    "کافی": "enough",
    "لیے": "for",
    "لکیر": "line",
    "سیدھی": "straight",
    "ٹیڑھی": "curved",
    "تشریف": "presence",
    "لائے": "brought",
    "دی": "given",
    "گئی": "was",
    "تاریخ": "date",
    "بینائی": "vision",
    "دھندلا": "blurred",
    "پن": "ness",
    "تھیک": "okay",
    "تھی": "was (she)",
    "جیسے": "like",
    "مرضی": "choice",
    "نم": "moist",
    "رکھیں": "keep",
    "چلو": "let's go",
    "ریستوراں": "restaurant",
    "ریاضی": "math",
    "میرے": "my",
    "مشکل": "difficult",
    "موضوع": "topic",
    "کاروباری": "business",
    "لےسکتا": "can take",
    "شاید": "maybe",
    "بارش": "rain",
    "گی": "will (she)",
    "بھائی": "brother",
    "چھوٹا": "small",
    "انٹرنیٹ": "internet",
    "کنکشن": "connection",
    "سست": "slow",
    "نام": "name",
    "میری": "my",
    "پہلے": "before",
    "بہتر": "better",
    "ہوگئی": "became",
    "نقد": "cash",
    "ادائیگی": "payment",
    "ضروری": "necessary",
    "اگلی": "next",
    "ساتھ": "with",
    "خوشی": "happiness",
    "ہوئی": "happened",
    "امتحان": "exam",
    "باتیں": "talks",
    "نہ": "not",
    "او": "oh",
    "پی": "drink something",
    "ریفریکشن": "refraction",
    "ڈیپارٹمنٹ": "department",
    "چھوٹی": "small",
    "عمر": "age",
    "آپریشن": "operation",
    "خون": "blood",
    "ٹیسٹ": "test",
    "شعبہ": "department",
    "امراض": "diseases",
    "بچکان": "pediatric",
    "براہ": "via",
    "مہربانی": "kindness",
    "باقاعدہ": "regular",
    "چیک": "check",
    "ملتے": "meet",
    "کرکےآہستہ": "doing slowly",
    "آہستہ": "slowly",
    "اشارہ": "signal",
    "شہر": "city",
    "ساٹھ": "sixty",
    "ہزار": "thousand",
    "لوگ": "people",
    "رہتے": "live",
    "تمباکو": "tobacco",
    "نوشی": "smoking",
    "یہیں": "here",
    "ٹھریئے": "stay",
    "یہان": "here",
    "سستے": "cheap",
    "دوسری": "second",
    "لیبارٹریوں": "laboratories",
    "مہنگے": "expensive",
    "شکریہ": "thank you",
    "ہمارے": "our",
    "کھانا": "food",
    "کھانے": "eating",
    "کیک": "cake",
    "تندور": "oven",
    "جھلّی": "membrane",
    "صاف": "clean",
    "پرواز": "flight",
    "گھنٹے": "hours",
    "تاخیر": "delay",
    "کوڑے": "trash",
    "ٹرک": "truck",
    "ہفتے": "week",
    "آتا": "comes",
    "پولیس": "police",
    "اسٹیشن": "station",
    "دو": "two",
    "بلاکس": "blocks",
    "پردہ": "curtain",
    "بصارت": "vision",
    "استاد": "teacher",
    "بہت": "very",
    "سوالات": "questions",
    "پوچھے": "asked",
    "حادثہ": "accident",
    "انگلیوں": "fingers",
    "ہجے": "spell",
    "گر پڑے": "fall",
    "اہم": "important",
    "سوتی": "cotton",
    "جرابوں": "socks",
    "جرابیں": "socks",
    "پہننے": "wearing",
    "کریم": "cream",
    "خوراک": "food",
    "چارٹ": "chart",
    "آدھا": "half",
    "انچ": "inch",
    "تلوے": "soles",
    "والے": "ones",
    "سادہ": "simple",
    "جوتے": "shoes",
    "سیلف": "self",
    "مینجمنٹ": "management",
    "گلوکوز": "glucose",
    "حوالہ": "reference",
    "ہم": "we",
    "سارے": "all",
    "اخراجات": "expenses",
    "برداشت": "bear",
    "مالی": "financial",
    "تعاون": "cooperation",
    "خوش": "happy",
    "آمدید": "welcome",
    "فلاحی": "welfare",
    "امداد": "aid",
    "رقم": "money",
    "کس": "who",
    "موجودہ": "current",
    "چشمے": "glasses",
    "شیشوں": "lenses",
    "حالت": "condition",
    "میل": "mail",
    "ایڈرس": "address",
    "والد": "father",
    "شوہر": "husband",
    "ماہانہ": "monthly",
    "آمدن": "income",
    "تنخواہ": "salary",
    "ٹیلیفون": "telephone",
    "نمبر": "number",
    "ہوئی": "happened",
    "رپورٹ": "report",
    "کونسا": "which",
    "کون": "who",
    "کونسی": "which",
    "دوائی": "medicine",
    "کیوں": "why",
    "واہ": "wow",
    "ہاں": "yes",
    "سفید": "white",
    "موتیا": "cataract",
    "نارمل": "normal",
    "سائز": "size",
    "بڑی": "big",
    
}


nlp = stanza.Pipeline("ur", model_dir="./stanza_resources")


def get_psl_sequence(urdu_sentence):
    doc = nlp(urdu_sentence)
    words = []
    pos_tags = []

    for sentence in doc.sentences:
        for word in sentence.words:
            words.append(word.text)
            pos_tags.append(word.upos)

    subject = []  # PRON
    time = []     # Temporal nouns or adverbs (e.g., کل)
    activity = [] # NOUN, ADJ, DET
    action = []   # VERB, AUX
    context = []  # NUM, ADP, others (additional modifiers)
    negation = []

    # Categorize words based on POS tags
    for i, word in enumerate(words):
        tag = pos_tags[i]
        if tag == "PRON":
            subject.append(word)
        elif tag in ["NOUN", "ADV"] and word in ["کل", "آج", "پرسوں"]:  #time 
            time.append(word)
        elif tag in ["NOUN", "ADJ", "DET"]:
            activity.append(word)
        elif tag in ["VERB"]:
            action.append(word)
        elif tag in ["NUM", "ADP"]:  # Numbers or context words
            context.append(word)
        elif tag in ["PART"]:
            negation.append(word)

    # Create the initial PSL sequence
    psl_sequence = subject + time + activity + context + action + negation

    # Add question mark at the end if needed
    if "؟" in urdu_sentence:
        psl_sequence.append("؟")
    
    question_list = ["کیا", "کون","کہاں","کیوں","کب","کیسے","کتنا","کس","کدھر","کتنی"]
    if any(word in question_list for word in urdu_sentence):
        psl_sequence.append("؟")


    # Special case: Remove leading "کیا" for PSL grammar
    if psl_sequence and psl_sequence[0] == "کیا":
        psl_sequence.pop(0)

    # Insert "یا" in the PSL sequence before the same word it precedes in the Urdu sentence
    if "یا" in urdu_sentence:
        urdu_words = urdu_sentence.split()  # Split the Urdu sentence into words
        for i, word in enumerate(urdu_words):
            if word == "یا":  # Check if the current word is "یا"
                next_word = urdu_words[i + 1]  # Get the word after "یا" in the Urdu sentence
                if next_word in psl_sequence:
                    position = psl_sequence.index(next_word)  # Find its position in the PSL sequence
                    psl_sequence.insert(position, "یا")  # Insert "یا" immediately before it

    # Define the stop words
    stop_words = set(["ہے", "اور", "سے", "کی", "کہ","کے", "کا", "ہیں", "پر", "کو", "نے", "جی", "پہ", "کی", "اس"])

    # Remove stop words from the PSL sequence
    cleaned_psl_sequence = [word for word in psl_sequence if word not in stop_words]
    return cleaned_psl_sequence

# function to  load gloss dictionary from the csv file
def load_gloss_dictionary(file_path):
    dataset = pd.read_csv(file_path, encoding="utf-8")
    return dict(zip(dataset["Word"], dataset["Gloss"]))


# for english to urdu in my gloss 
def translate_urdu_to_english(urdu_words):
    tokenized_text = tokenizer(urdu_words, return_tensors="pt", padding = True , truncation= True)
    # print(f"tokenizes word: {tokenized_text}")
    translations = model.generate(**tokenized_text)
    # print(f"translation: {translations}" )
    english_words = [tokenizer.decode(t, skip_special_tokens=True).lower() for t in translations]
    return english_words

# print(translate_urdu_to_english("تین ماہ بعد مسلمان گھر میں رہتے ہیں"))

def map_sentence_to_gloss(urdu_sentence):
    print("------------------------------------------------------------------------------")
    print("Urdu Sentence: ", urdu_sentence)
    words = urdu_sentence.split()  #original urdu words from sentences
    print(f'Urdu sentence after split: ', words)
    stop_words = set(["ہے", "اور", "سے", "کی", "کہ","کے", "کا", "ہیں", "پر", "کو", "نے", "جی", "پہ", "کی", "اس"])
    words = [word for word in words if word not in stop_words]

    # Translate each word to English
    # english words created form translating urtdu words
    english_words = translate_urdu_to_english(words)  
    print(f'MarianMT translated each urdu word to english : ', english_words)
    
    gloss_sentence = []  # Final list to store glosses for the sentence

    for urdu_word, english_word in zip(words, english_words):
        # getting the english mapping for urdu word in urdu words list from sentence
        gloss_key = gloss_dictionary.get(urdu_word)
        # print(f'Gloss key (english words) generated from dictionary by get function: ', gloss_key)

        """If no direct match, look for a match based on meaning (check English translation) like no match for word abbu then look for english translation for
         word abbu and it is father then extract key of that fathr which ultimately give the word walid
         by this (below) we have only those words in wnglish which are present in our gloss dictionary"""
        if gloss_key is None:
            # Find a key in the dictionary that matches the English translation
            key = next((key for key, value in gloss_dictionary.items() if value == english_word), None)
            # print(f'key ', key)
            
            if key:  # If a valid key is found
                gloss_key = gloss_dictionary[key]
                # print(f'gloss key ', gloss_key)
                gloss_sentence.append(gloss_key) # urdu word found
            else:
                # If no key is found, append the original Urdu word or handle as needed
                gloss_sentence.append(f"{urdu_word}")
        else:
            gloss_sentence.append(gloss_key)
        
    # Return the final gloss sentence
    # return " ".join(gloss_sentence)
    return gloss_sentence


# Test the function

def is_urdu_word(word):
    return any("\u0600" <= char <= "\u06FF" for char in word)


gloss_dictionary_path = "word_gloss_dataset.csv"
gloss_dictionary = load_gloss_dictionary(gloss_dictionary_path)
output_dir = "./marian-finetuned"
model = MarianMTModel.from_pretrained(output_dir)
tokenizer = MarianTokenizer.from_pretrained(output_dir)


@app.route('/generate-psl', methods =['POST'])
def generate_psl():
    try:
        urdu_sentence = request.json.get("urdu_text","")
        if not urdu_sentence:
            return jsonify({"error": "No urdu text provided"}), 400
        gloss_output = map_sentence_to_gloss(urdu_sentence)
        # gloss_output1 = map_sentence_to_gloss(urdu_sentence1)
        

        urdu_word_list1 = []
        # urdu_word_list2 = []
        for word in gloss_output:
            if is_urdu_word(word):
                urdu_word_list1.append(word)
            else:
                key = next((key for key, value in gloss_dictionary.items() if value == word), None)
                print(key)
                urdu_word_list1.append(key if key else "noword")



        urdu_word_list1 = " ".join(urdu_word_list1)
        final_spl = get_psl_sequence(urdu_word_list1)

        # print(f"Urdu Sentence: {urdu_sentence}")
        print(f"Mapped Gloss Sentence: {gloss_output}")
        #below is elimnating the words that dont need in sign perfomance

        print("Before Applying text-Gloss mapping Rules : ", urdu_word_list1)
        print(f"Final Psl Sequence : {final_spl}")
        print("------------------------------------------------------------------------------")
        final_spl = " ".join(final_spl)
        return jsonify({'final_psl': final_spl})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# if __name__ == "__main__":
#     app.run(port=5001, debug=True)
if __name__ == "__main__":
    app.run(debug=True)