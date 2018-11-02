# DivideMusicGenre
 It expresses music by MFCC and makes machine learning its value.You can use it to classify music genres and to distinguish sounds.However, there are a couple of things to change and some points to be aware of when making models.

# Dependency
 ## installing "librosa"
 ~~~
 $ pip install librosa
 ~~~

# using
 ## file structure
 It is recommended that the file structure be as follows.
 ~~~
 DivideMusicGenre ──┬── DivideMusicGenre.ipynb
                    │
                    └── wav(directory) ──┬── MusicGenreA (directory)
                                         ├── MusicGenreB (directory)
                                         └── MisicGenreC (directory)
 ~~~

 It would be nice to prepare directories for each genre and put wav files of each genre in nearly the same number in that directory. In the example above, there are three genres, A, B, and C. However, this time we only have two genres in this Github, "yes" and "no". "yes" means that the wav file in "yes"directory a whistle, and "no" means that the wav file in "yes"directory a whistle.

 ## changing genre
 If you want to change number of genre/name of genre, you must change part of the source code. Now, I want change number of genre from 2 (yes/no) to 3 (A/B/C).In that case, change it as follows.

 ~~~python3
 '''
 # it is a old source code.
 yes_x, yes_y = load('wav/yes/', 'yes')
 no_x, no_y = load('wav/no/', 'no')

 X = np.r_[yes_x, no_x]
 Y = np.r_[yes_y, no_y]
 '''

 # it is a new source code.
 A_x, A_y = load('wav/MusicGenreA/', 'MusicGenreA')
 B_x, B_y = load('wav/MusicGenreB/', 'MusicGenreB')
 C_x, C_y = load('wav/MusicGenreC/', 'MusicGenreC')

 X = np.r_[A_x, B_x, C_x]
 Y = np.r_[A_y, B_y, C_y]
 ~~~
