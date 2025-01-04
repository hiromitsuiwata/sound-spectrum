# sound-spectrum

## 仮想環境の構築

```bash
python3 -m venv .venv
source .venv/bin/activate
sudo apt install portaudio19-dev
pip install jupyterlab numpy matplotlib scipy scikit-image wave  portaudio19-dev
jupyter lab
```

## 参考

- [Pythonでスペクトル解析【音声認識実践その1】](https://www.itd-blog.jp/entry/voice-recognition-10)
- [【python】自己相関関数を用いたピッチ検出【サウンドプログラミング】](https://ism1000ch.hatenablog.com/entry/2014/08/27/015052)
- [音声の波形からピッチを検出するアルゴリズム](https://mametter.hatenablog.com/entry/20120122/p1)
- [File:Kimiko Ishizaka - Bach - Well-Tempered Clavier, Book 1 - 01 Prelude No. 1 in C major, BWV 846.ogg](https://en.wikipedia.org/wiki/File:Kimiko_Ishizaka_-_Bach_-_Well-Tempered_Clavier,_Book_1_-_01_Prelude_No._1_in_C_major,_BWV_846.ogg)
- [Python リアルタイム音声処理: pydubとPyAudioの組み合わせ](https://qiita.com/Tadataka_Takahashi/items/e30da1d30e4dc2e255d1)
