pip install -r requirements.txt

#download Waveglow
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

#download mels
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
mkdir data
tar -xvf mel.tar.gz >> data
rm mel.tar.gz

#download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
mkdir -p data/alignments
unzip alignments.zip >> data/alignments
rm alignments.zip

# we will use waveglow code, data and audio preprocessing from this repo
git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/text .
mv FastSpeech/audio .
mv FastSpeech/waveglow/* waveglow/
mv FastSpeech/utils.py .
mv FastSpeech/glow.py .
rm -r FastSpeech