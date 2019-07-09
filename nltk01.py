#!/usr/bin/env python
text = ( " To suppose that the eye with all its inimitable contrivances "
" for adjusting the focus to different distances , for admitting "
" different amounts of light , and for the correction of spherical "
" and chromatic aberration , could have been formed by natural "
" selection , seems , I freely confess , absurd in the highest degree . "
" When it was first said that the sun stood still and the world "
" turned round , the common sense of mankind declared the doctrine "
" false ; but the old saying of Vox populi , vox Dei , as every "
" philosopher knows , cannot be trusted in science . " )

import nltk
nltk.download('punkt')
pos_tags = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(text)]
print(pos_tags[0][:5])
word_tags=[ word for sent in pos_tags for word , tag in sent if tag =='NN'] # Nouns
print(word_tags)
