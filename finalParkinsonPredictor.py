import joblib
from sklearn.tree import DecisionTreeClassifier
import speech_recognition as sr
from numpy import size
import pydub
from pydub import AudioSegment
from aubio import source, pitch
import parselmouth
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score 
import wave
import aubio
import librosa

FILE_NAME="output12"+".wav"

def record():

    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)

        print("say something")

        audio = r.listen(source)
        print("Recognizing the sentence")


        try:
            text=r.recognize_google(audio)
            #print("User has said \n" + text)
            words=text.split()
            print("Your voice has been recorded Successfully. \n ")
            print("Please wait... processing your data\n")


        except Exception as e:
            print("Error :  " + str(e))


        with open(FILE_NAME, "wb") as file:
            file.write(audio.get_wav_data())

        return words

words=record()


audio_file = AudioSegment.from_wav(FILE_NAME)


windowSize = 4096
hopSize = 512
samplerate = audio_file.frame_rate


s = source(FILE_NAME, samplerate, hopSize)
p = pitch("yin", windowSize, hopSize, samplerate)


pitchList = []
pitch_max_list = []

total_frames = 0
while True:
    samples, read = s()

    pitchValue = p(samples)[0]

    if pitchValue > 100 and pitchValue<700:
        pitchList.append(pitchValue)

    total_frames += read
    if read < hopSize: 
        break


minPitch = min(pitchList)
maxPitch = max(pitchList)
avgPitch = sum(pitchList) / len(pitchList)


snd = parselmouth.Sound(FILE_NAME)

harmonics = snd.to_harmonicity()


amplitude = snd.to_intensity()
meanAmplitude = np.mean(amplitude.values.T)
shimmerPercent = 100 * np.mean(np.abs(amplitude.values.T[1:] - amplitude.values.T[:-1])) / meanAmplitude
shimmerAbsolute = np.mean(np.abs(amplitude.values.T[1:] - amplitude.values.T[:-1]))


differences = np.abs(amplitude.values.T[1:] - amplitude.values.T[:-1])
shimmerAPQ = np.mean(differences / np.mean(np.array([amplitude.values.T[1:], amplitude.values.T[:-1]]), axis=0)) 


signal, sr = librosa.load(FILE_NAME, sr=44100)

hopSize = 256
pitchObj = aubio.pitch("default", 2048, hopSize, sr)
source = aubio.source(FILE_NAME, sr, hopSize)


fundamentalFreq = []
while True:
    samples, read = source()
    if read < hopSize:
        break
    pitch = pitchObj(samples)[0]
    fundamentalFreq.append(pitch)



jitterAbsolute = np.abs(np.diff(fundamentalFreq))
jitterDecimal = jitterAbsolute / np.mean(fundamentalFreq[:-1])



model = joblib.load('svm_model.pkl')

input_data=(avgPitch,maxPitch,minPitch,jitterDecimal,jitterAbsolute,shimmerPercent,shimmerAbsolute,shimmerAPQ)


input_data_as_numpy_array= np.asarray(input_data)

input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
scaler=StandardScaler()
std_data= scaler.transform(input_data_reshaped)

prediction1= model.predict(std_data)
print(prediction1)

predict1=False
if(prediction1[0]==0):
    print("person is healthy")
else:
    print("person is unhealthy")
    predict1=True



with wave.open(FILE_NAME, 'rb') as wav:
    frame_rate = wav.getframerate()
    num_frames = wav.getnframes()
    duration = num_frames / float(frame_rate)
wordOffset=duration/len(words)


wordOffsetClassifier = joblib.load('parkinsons_word_offset_model.joblib')

testList=[[wordOffset]]

prediction2=wordOffsetClassifier.predict(testList)
print(prediction2)
predict2=False
if(prediction2[0]==0):
    print("person is healthy")
else:
    print("person is unhealthy")
    predict2=True

if predict1 and predict2:
    print("The patient is at risk or may be suffering from Parkinson's disease")
else:
    print("The person is not at risk from Parkinson's disease")
  

