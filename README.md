# DivideMusicGenre
 It expresses music by MFCC and makes machine learning its value.You can use it to classify music genres and to distinguish sounds.However, there are a couple of things to change and some points to be aware of when making models.

# Dependency
 ## installing "librosa"
 ~~~
 $ pip install librosa
 ~~~

# Usage
 ## file structure
 It is recommended that the file structure be as follows.
 ~~~
 DivideMusicGenre/
        ├── DivideMusicGenre.ipynb
        ├── MakeModels.py
        ├── Judgment.py
        ├── testWav/
        │    ├── answer0_1.wav
        │          ....
        └── wav/
             ├── MusicGenreA/
             ├── MusicGenreB/
             └── MisicGenreC/
 ~~~

 It would be nice to prepare directories for each genre and put wav files of each genre in nearly the same number in that directory. In the example above, there are three genres, A, B, and C. However, this time we only have two genres in this Github, "yes" and "no". "yes" means that the wav file in "yes"directory a whistle, and "no" means that the wav file in "yes"directory a whistle.

 ## changing genre
 If you want to change number of genre/name of genre, you must change part of the source code. Now, I want change number of genre from 2 (yes/no) to 3 (A/B/C).In that case, change it as follows.

 ~~~python
 # MakeModel.py
 '''
 # it is a old source code.
 yes_x, yes_y = load('wav/yes/', '0')
 no_x, no_y = load('wav/no/', '1')

 X = np.r_[yes_x, no_x]
 Y = np.r_[yes_y, no_y]
 '''

 # it is a new source code.
 A_x, A_y = load('wav/MusicGenreA/', '0')
 B_x, B_y = load('wav/MusicGenreB/', '1')
 C_x, C_y = load('wav/MusicGenreC/', '2')

 X = np.r_[A_x, B_x, C_x]
 Y = np.r_[A_y, B_y, C_y]
 ~~~
 The second argument of the ```load``` function must be a number enclosed with Quarte Shooting mark. 

 ## Let's test!

 ~~~python
 # Judgment.py
 from keras.models import load_model

 model = load_model('model_whistle.h5')
 test_mcff = load_a_file('/testWav/answer0_1.wav') #you can change file path
     
 test_mcff_final = test_mcff.reshape(1, 20, 11, 1)
 print(np.argmax(model.predict(test_mcff_final)))
 ~~~
 Place the test wav file in the ```testWav``` directory and the file path with relative path. The label name is displayed. (Such as 0 or 1)

 ## GoogleColaboratory
 I give you DivideMusicGenre.ipynb. This can divide music in GoogleColaboratory.Please mount GoogleDrive and execute the program. It is recommended that the file structure be as follows in GoogleDrive.
 ~~~
 GoogleDrive/
   └── Colab Notebook/
             ├── DivideMusicGenre.ipynb
             ├── wav/
             │    ├── MusicGenreA/
             │    ├── MusicGenreB/
             │    └── MisicGenreC/
             └── testWav/
                    ├── answer0_1.wav
                    　　　 ....
 ~~~

 When you mount it, it looks like this in Google Colaboratory.
 ~~~
My Drive/
    └── Drive/
          └── Colab Notebook/
                   ├── DivideMusicGenre.ipynb
                   ├── wav/
                   │    ├── MusicGenreA/
                   │    ├── MusicGenreB/
                   │    └── MisicGenreC/
                   └── testWav/
                        ├── answer0_1.wav
                       　　　 ....
 ~~~
# Licence
 This software is released under the ***BSD 3-Clause License***, see LICENSE.