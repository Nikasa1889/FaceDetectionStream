from gtts import gTTS
from time import sleep
import os
import pyglet
import sys

def say(language, text):
  tts = gTTS(text=text, lang=language)
  filename = 'temp.mp3'
  tts.save(filename)

  music = pyglet.media.load(filename, streaming=False)
  music.play()

  sleep(music.duration) #prevent from killing
  os.remove(filename) #remove temperory file

if __name__ == "__main__":
  say(sys.argv[1], sys.argv[2])
