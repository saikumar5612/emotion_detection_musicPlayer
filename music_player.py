import tkinter as tk
from tkinter import filedialog
import pygame
import os

# Initialize Pygame mixer
pygame.mixer.init()

# Emotion-specific music files
music_library = {
    "happy": "music/happy_song.mp3",
    "sad": "music/sad_song.mp3",
    "angry": "music/angry_song.mp3",
    "neutral": "music/neutral_song.mp3"
}

class MusicPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Music Player")

        # Play button
        self.play_btn = tk.Button(root, text="Play", command=self.play_music)
        self.play_btn.pack()

    def play_music(self, emotion="neutral"):
        music_file = music_library.get(emotion, music_library["neutral"])
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()

if __name__ == '__main__':
    root = tk.Tk()
    app = MusicPlayer(root)
    root.mainloop()
