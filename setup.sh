echo "Install requirements"
pip install -r requirements.txt

echo "Download train texts"
gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mkdir data
mv train.txt data/

echo "Download mels"
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz -C data >> /dev/null
rm mel.tar.gz

echo "Download pitches"
gdown https://drive.google.com/uc?id=1JdeYK7xm_ABaJZtc7CQiBdHSQnFGPA9F
tar -xvf pitches.tar -C data >> /dev/null
rm pitches.tar

echo "Download alignments"
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip -d data >> /dev/null
rm alignments.zip

echo "Download Waveglow checkpoint"
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

echo "Clone FastSpeech modules"
git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/text .
mv FastSpeech/waveglow/* waveglow/
mv FastSpeech/utils.py .
mv FastSpeech/glow.py .
mv FastSpeech/hparams.py .
rm -rf FastSpeech
