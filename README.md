# TTS project

## Installation guide

```shell
pip install -r ./requirements.txt
mkdir lm
wget https://www.openslr.org/resources/11/3-gram.arpa.gz -P lm
gzip -d lm/3-gram.arpa.gz
```
## Train running guide
Add your model config to `hw_asr/configs/` and run:

```shell
python3 train.py -c hw_tts/configs/your_config.json
```
In order to recreate results, use `train_jasper.json`:
```shell
python3 train.py -c hw_tts/configs/train_jasper.json
```
By default, config assumes that it is used in kaggle with [librispeech](https://www.kaggle.com/datasets/a24998667/librispeech/). If you use other trainig sources, you may want to change `data_dir` path in config.
## Test running guide
First of all, download model checkpoint and its config:
```shell
cd default_test_model
wget "https://www.dropbox.com/scl/fi/10oj65gl4w66ij9b4yu5o/config.json?rlkey=teanfhvo8ppcwqxudj8cgn8m6&dl=0" -O config.json
wget "https://www.dropbox.com/scl/fi/coj8d16hcf4og27nhxe9d/model_best-2.pth?rlkey=l88vetlrhlfomuhwjdk1idmn0&dl=0" -O checkpoint.pth
cd ..
```
Run test-clean
```shell
python3 test.py \
   -c default_test_model/test_lm_jasper_clean.json \
   -r default_test_model/checkpoint.pth \
   -o test_result_clean.json
```
After running test, test_result_clean.json file should be created. All metrics would be written at the end of the file.

Also, you can run test on test-other using `test_lm_jasper_other.json` or on your own data with option `-t`.
