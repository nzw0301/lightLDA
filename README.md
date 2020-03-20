# LightLDA.py

This repo is python reimplementation of [lightLDA](https://github.com/Microsoft/LightLDA).

LightLDA is a topic scalable latent dirichlet allocation (LDA) algorithm that is proposed in [WWW paper](http://www.www2015.it/documents/proceedings/proceedings/p1351.pdf).

## Examples

```bash
$ python lightlda

# Word distributions per latent class
## φ of latent class of 0
word: probability
安原絵麻: 0.346
SHIROBAKO: 0.266
万策尽きた: 0.161
佳村はるか: 0.108
武蔵野: 0.108
城ヶ崎美嘉: 0.003
デレマス: 0.003
城ヶ崎莉嘉: 0.003
カブトムシ: 0.003

## φ of latent class of 1
word: probability
城ヶ崎美嘉: 0.357
デレマス: 0.239
佳村はるか: 0.180
城ヶ崎莉嘉: 0.121
カブトムシ: 0.091
安原絵麻: 0.003
SHIROBAKO: 0.003
万策尽きた: 0.003
武蔵野: 0.003

# Topic distributions per document
## Topic information of document 0
Propotion of topics
topic: θ_{document_id, latent_class}
0: 0.001
1: 0.999

Assigned latent class per word
word: latent class
城ヶ崎美嘉: 1
城ヶ崎美嘉: 1
城ヶ崎美嘉: 1
城ヶ崎美嘉: 1
デレマス: 1
デレマス: 1
佳村はるか: 1
佳村はるか: 1
佳村はるか: 1
--------------

## Topic information of document 1
Propotion of topics
topic: θ_{document_id, latent_class}
0: 0.001
1: 0.999

Assigned latent class per word
word: latent class
城ヶ崎美嘉: 1
城ヶ崎美嘉: 1
城ヶ崎美嘉: 1
城ヶ崎美嘉: 1
城ヶ崎美嘉: 1
城ヶ崎美嘉: 1
佳村はるか: 1
デレマス: 1
デレマス: 1
城ヶ崎莉嘉: 1
城ヶ崎莉嘉: 1
カブトムシ: 1
--------------

## Topic information of document 2
Propotion of topics
topic: θ_{document_id, latent_class}
0: 0.001
1: 0.999

Assigned latent class per word
word: latent class
城ヶ崎美嘉: 1
城ヶ崎美嘉: 1
佳村はるか: 1
佳村はるか: 1
デレマス: 1
デレマス: 1
デレマス: 1
デレマス: 1
城ヶ崎莉嘉: 1
城ヶ崎莉嘉: 1
カブトムシ: 1
カブトムシ: 1
--------------

## Topic information of document 3
Propotion of topics
topic: θ_{document_id, latent_class}
0: 0.999
1: 0.001

Assigned latent class per word
word: latent class
安原絵麻: 0
安原絵麻: 0
安原絵麻: 0
佳村はるか: 0
佳村はるか: 0
SHIROBAKO: 0
SHIROBAKO: 0
万策尽きた: 0
万策尽きた: 0
--------------

## Topic information of document 4
Propotion of topics
topic: θ_{document_id, latent_class}
0: 0.999
1: 0.001

Assigned latent class per word
word: latent class
安原絵麻: 0
安原絵麻: 0
安原絵麻: 0
佳村はるか: 0
SHIROBAKO: 0
SHIROBAKO: 0
武蔵野: 0
武蔵野: 0
万策尽きた: 0
--------------

## Topic information of document 5
Propotion of topics
topic: θ_{document_id, latent_class}
0: 0.999
1: 0.001

Assigned latent class per word
word: latent class
安原絵麻: 0
安原絵麻: 0
安原絵麻: 0
安原絵麻: 0
安原絵麻: 0
安原絵麻: 0
安原絵麻: 0
佳村はるか: 0
SHIROBAKO: 0
SHIROBAKO: 0
SHIROBAKO: 0
SHIROBAKO: 0
SHIROBAKO: 0
SHIROBAKO: 0
万策尽きた: 0
万策尽きた: 0
万策尽きた: 0
武蔵野: 0
武蔵野: 0
--------------
```

## Reference

Yuan, Jinhui and Gao, Fei and Ho, Qirong and Dai, Wei and Wei, Jinliang and Zheng, Xun and Xing, Eric Po and Liu, Tie-Yan and Ma, Wei-Ying. LightLDA: Big Topic Models on Modest Computer Clusters.
 In _WWW_, 2015.
