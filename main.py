#Imports

## General Imports
import numpy as np
import math

## Visualization
import matplotlib.pyplot as plt
import IPython.display as ipd
from ipywidgets import interactive_output 
from ipywidgets import IntSlider, FloatSlider, fixed, Checkbox
from ipywidgets import VBox, Label
import pygame

## Audio Imports
import librosa, librosa.display  
from music21.tempo import MetronomeMark  
from music21.note import Note, Rest
from music21.stream import Stream
from music21 import metadata
from music21 import instrument
from music21 import environment
import mutagen

## Matplotlib
plt.rc("figure", figsize=(16, 8))

# Load Audio (play audio and check with user)
done = "no"
while done == "no":
  filename = input(
    "Please input the folder and the location of the song you want to transform. For example: ./Music/Mary.mp3 "
  )
  #Change this for each file
  filename = filename
  #name = 'Mary.mp3.mp3'
  #playback audio file and confirm
  pygame.init()
  pygame.mixer.music.load(filename)

  #make sure it is the correct audio file
  while True:
    pygame.mixer.music.play()
    ans = input("Is this your song?\n Type yes or no.")
    if ans == "yes":
      done = "yes"
    pygame.mixer.pause()
    break

#determining sampling rate of audio file
audio_info = mutagen.File(filename).info
fs = audio_info.sample_rate
#print("Sampling rate", fs)

#duration of audio file in seconds
dur = audio_info.length
#print(dur)

# Parameters
## Signal Processing 
nfft = 2048  # length of the FFT window (for stable frequencies at 100Hz)
overlap = 0.5  # Hop overlap percentage (standard 50%)
hop_length = int(nfft *(1 - overlap))  # Number of samples between successive frames
n_bins = 72  # Number of frequency bins
mag_exp = 4  # Magnitude Exponent 
pre_post_max = 6  # Pre- and post- samples for peak picking
cqt_threshold = -60  # Threshold for CQT dB levels, all values below threshold are set to -120 dB (threshold of -50) might make trouble because of dynamics

songname = input("What is the title of the song? ")
sn = metadata.Metadata(title = songname)

composer = input("Who is the composer/artist? ")
comp = metadata.Contributor(role = 'composor', name = composer)

#sr = sampling rate given in frequency (None = native sampling rate of 22050) mono = convert signal to mono duration = only load up to this much audio
# x = np.ndarray audio time series and fs = sampling rate of x
x, fs = librosa.load(filename, sr=None, mono=True, duration=dur)

# CQT
# Function
#CQT TRANSFORMS A DATA SERIES TO THE FREQUENCY DOMAIN
#x = audio time series, fs = 22050 (sampling rate), hop_length = number of samples between successive CQT columns, n_bins = number of frequency bins per octave, mag_exp = phase components)
def calc_cqt(x, fs=fs, hop_length=hop_length, n_bins=n_bins, mag_exp=mag_exp):
  #returns the constant q value of each frequency at each time in an array
  C = librosa.cqt(x, sr=fs, hop_length=hop_length, fmin=None, n_bins=n_bins)
  #separates a complex-valued spectrogram D into its magnitude (S) and phase (P) components so that D = S * P (outputs an amplitude)
  C_mag = librosa.magphase(C)[0]**mag_exp
  #converts an amplitude spectrogram into dB-scaled spectrogram
  CdB = librosa.core.amplitude_to_db(C_mag, ref=np.max)
  #returns array measured in dB
  return CdB


# CQT Threshold (getting rid of outside noise)
#takes all the background noise that is under a certain threshold and makes them -120dB
def cqt_thres(cqt, thres = cqt_threshold):
  new_cqt = np.copy(cqt)
  new_cqt[new_cqt < thres] = -120
  return new_cqt


#convert the transformed audio file into music sheet
# Onset Envelope from Cqt
#determining when the note starts and ends (timing of each note)
#Computes a spectral flux onset strength envelope. "Onset strength at time, t determined by mean_f max(0, S[f/t]-ref[f,t-lag])"
#S = pre-computed (log-power) spectrogram, sr = sampling rate, aggregate = aggregation function to use when combining onsets at different frequency bins, hop_length = number of samples between successive CQT columns
#returns: vector containing the onset strength envelope (starting to ending time of the note)
def calc_onset_env(cqt):
  return librosa.onset.onset_strength(S=cqt,
                                      sr=fs,
                                      aggregate = np.mean,
                                      hop_length=hop_length)


#Locates note onset events by picking peaks in an onset strength envelope
def calc_onset(cqt, pre_post_max=pre_post_max, backtrack=True):
  onset_env = calc_onset_env(cqt)
  #returns an estimated position of detected onsets, in frame indices
  onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,
                                            sr=fs,
                                            units='frames',
                                            hop_length=hop_length,
                                            backtrack=backtrack,
                                            pre_max=pre_post_max,
                                            post_max=pre_post_max)
  #joins a sequence of arrays along an existing axis
  onset_boundaries = np.concatenate([[0], onset_frames, [cqt.shape[1]]])
  #converts frame counts to time (seconds)
  #returns time (in seconds) of each given frame number
  onset_times = librosa.frames_to_time(onset_boundaries,
                                       sr=fs,
                                       hop_length=hop_length)
  return [onset_times, onset_boundaries, onset_env]


# Fine Tuning UI
#create a visual representation of cqt and each frequency
style = {'description_width': 'initial'}
mag_exp_slider = IntSlider(value=mag_exp,
                           min=1,
                           max=32,
                           step=1,
                           description='mag_exp:',
                           continuous_update=False)

thres_slider = IntSlider(value=cqt_threshold,
                         min=-120,
                         max=0,
                         step=1,
                         description='Threshold:',
                         continuous_update=False)

pre_post_slider = IntSlider(value=pre_post_max,
                            min=1,
                            max=32,
                            step=1,
                            description='Pre_post_max:',
                            continuous_update=False,
                            style=style)

backtrack_box = Checkbox(value=False, description='backtrack', disabled=False)


def inter_cqt_tuning(mag_exp, thres, pre_post_max, backtrack):
  thres = thres_slider.value
  mag_exp = mag_exp_slider.value
  pre_post_max = pre_post_slider.value
  backtrack = backtrack_box.value
  global CdB
  CdB = calc_cqt(x, fs, hop_length, n_bins, mag_exp)
  #create graph (matplotlib)
  plt.figure()
  new_cqt = cqt_thres(CdB, thres)
  librosa.display.specshow(new_cqt,
                           sr=fs,
                           hop_length=hop_length,
                           x_axis='time',
                           y_axis='cqt_note',
                           cmap='coolwarm')
  #range on y-axis
  #converts the note names to frequency (how determined?)
  plt.ylim([librosa.note_to_hz('B2'), librosa.note_to_hz('B6')])
  global onsets
  onsets = calc_onset(new_cqt, pre_post_max, backtrack)
  #vertical lines
  plt.vlines(onsets[0], 0, fs / 2, colors='k', linestyles='solid', alpha=0.8)
  plt.title(songname)
  plt.colorbar()
  plt.show()
  #print("Onsets= ", onsets)
  return onsets


print('This process may take a while... \n When finished adjusting, click on the "x" in the top right corner to continue.')

# Display UI
out = interactive_output(inter_cqt_tuning,  {'mag_exp': mag_exp_slider, 'thres': thres_slider,
                        'pre_post_max': pre_post_slider, 'backtrack':backtrack_box})
ui = VBox([mag_exp_slider, thres_slider, pre_post_slider, backtrack_box])

#print("Onsets=", onsets)

# Estimate Tempo
#user estimated guess of tempo
while True:
  try:
    usertempo = int(input("Enter your estimated tempo for the song in bpm: "))
    if usertempo <= 0:
      print("This is not a valid tempo.")
    else:
      break
  except:
    print("This is not a valid tempo.")


tempo, beats = librosa.beat.beat_track(y=None,
                                       sr=fs,
                                       onset_envelope=onsets[2],
                                       hop_length=hop_length,
                                       start_bpm=usertempo,
                                       tightness=100,
                                       trim=True,
                                       bpm=None,
                                       units='frames')
tempo = int(2 * round(tempo / 2))
#print("tempo= ", tempo)

#ask user tempo per what beat
referent = input("Enter the referring beat per minute (options: sixteenth, eighth, quarter, half, whole) ")
mm = MetronomeMark(referent = referent, number = tempo)

#print("metronome mark= ", mm)
# <music21.tempo.MetronomeMark maestoso Quarter=90>

#FOR MIDI
# Convert Seconds to Quarter-Notes
def time_to_beat(duration, tempo):
  beat = (tempo * duration / 60)
  #print("beat", beat)
  return beat

#music information array
# Generate Sinewave, MIDI Notes and music21 notes
def generate_note(f0_info, sr, n_duration, round_to_sixtenth=True):
  f0 = f0_info[0]
  
  #converts frame counts to time (seconds) for one note
  duration = librosa.frames_to_time(n_duration, sr=fs, hop_length=hop_length)
  #print("duration= ", duration)

  #Generate Midi Note and music21 note
  note_duration = math.ceil(duration*100)/100  # Round to 2 decimal places for music21 compatibility
  #print("note duration= ", note_duration)
  #midi_duration = time_to_beat(duration, tempo)
  #midi_velocity = int(round(remap(f0_info[1], CdB.min(), CdB.max(), 0, 127)))
  #print("midi_velocity= ", midi_velocity)
  #if round_to_sixtenth:
    #midi_duration = round(midi_duration * 16) / 16
  if f0 == None:
    #midi_note = None
    note_info = Rest(type=mm.secondsToDuration(note_duration).type)
    f0 = 0
  else:
    note = Note(librosa.hz_to_note(f0), type=mm.secondsToDuration(note_duration).type)
    #note.volume.velocity = midi_velocity
    #print("Type of note=", note.type)
    note_info = [note]
    #print(note_info)
  #midi_info = [midi_note, midi_duration, midi_velocity]

  # Generate Sinewave
  #print("note information= ", note_info)
  return note_info


#Estimate Pitch
#computes the center frequencies of cqt bins
def estimate_pitch(segment, threshold):
  freqs = librosa.cqt_frequencies(n_bins=n_bins,
                                  #lowest c on the piano
                                  fmin=librosa.note_to_hz('C1'),
                                  #chromatic scale
                                  bins_per_octave = 12)
  #print("Frequency of each note: ", freqs)
  #if the segment is quieter (background noise)
  if segment.max() < threshold:
    return [None, np.mean((np.amax(segment, axis=0)))]
  else:
  # determine center frequency (mean) of cqt for each note
    f0 = int(np.mean((np.argmax(segment, axis=0))))
    #print("freqs[f0]=", freqs[f0])
    #print("mean= ", np.mean((np.amax(segment, axis=0))))
    #print("estimate pitch= ", [freqs[f0], np.mean((np.amax(segment, axis=0)))], "\n done \n")
  return (freqs[f0], np.mean((np.amax(segment, axis=0))))

#print("hello")

# Generate notes from Pitch estimation
def estimate_pitch_and_notes(x, onset_boundaries, i, sr):
  #lower boundary (note start)
  n0 = onset_boundaries[i]
  #upper boundary (note end)
  n1 = onset_boundaries[i + 1]
  f0_info = estimate_pitch(np.mean(x[:, n0:n1], axis=1),
                           threshold=cqt_threshold)
  #print("HERE =", generate_note(f0_info, sr, n1 - n0))
  return generate_note(f0_info, sr, n1 - n0)

#print("hi")

# Array of music information - Sinewave, MIDI Notes and muisc21 Notes
#HEEEERRRRRREEEEEEEEEE

music_info = [estimate_pitch_and_notes(CdB, onsets[1], i, sr=fs) for i in range(len(onsets[1]) - 1)]

#print("music info= ", music_info)

# Create music21 stream
score = Stream()
# input metronome marking
score.append(mm)
#get user instrument 
instr = input("What is the intrument playing? \n")

instr = instrument.fromString(instr)

score.append(instr)

score.insert(0, metadata.Metadata())

score.metadata.title = songname
score.metadata.composer = composer
#print("music info so far=", s)

#put notes into score
for note in music_info:
  #print(note)
  score.append(note)


# Analyse music21 stream to get song Key
key = score.analyze('key')
#print(key.name)
# Insert Key to Stream
score.insert(0, key)

#print("score=", score)
us = environment.UserSettings()
us['showFormat'] = 'lilypond'
us['lilypondPath'] = 'c:./lilypond-2.25.5/bin/lilypond.exe'

# Display music21 stream
score.show() 

# Show stream as text
score.show('text')
