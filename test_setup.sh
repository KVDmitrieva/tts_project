echo "Install requirements"
pip install -r requirements.txt

echo "Download FastSpeech2 checkpoint"
gdown https://drive.google.com/u/0/uc?id=1ner68HZAbePhXd-vOclrhh0p1o6nKE9r
gdown https://drive.google.com/u/0/uc?id=1pg-7lmDrG_QF2u7gv8_EmseRSM2_-9_X
mv model_best.pth default_test_model/model.pth
mv fastspeech2.json default_test_model/config.json

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
