# STRAY

一镜到底的赛博朋克追车短片。城市、车、追兵和爆炸都在 Blender 里用代码搭出来，每一帧用 EEVEE 渲染。没有实拍，也没有视频生成模型。广告屏上的画是单独生成的静帧。

成片 64 秒，1080p，60 帧，16:9。配乐是 105 BPM 的原曲，28 个小节不间断；慢镜头只放慢雨、烟、灯和现场声，曲子本身不变速。

## 怎么重渲

需要 Blender 4.2 和 Python 3（numpy）。

```bash
# 节拍表和路程
python3 stray/timing.py

# 关键帧
blender -b -P stray/film.py -- --preview

# 全部 3840 帧，写到 build/frames/f_####.png
blender -b -P stray/film.py -- --render

# 曲子和现场声
python3 stray/music.py
python3 stray/mix_audio.py

ffmpeg -y -framerate 60 -i build/frames/f_%04d.png -i build/mix.wav \
  -vf "fade=t=out:st=62.3:d=1.7" \
  -c:v libx264 -pix_fmt yuv420p -crf 17 -tune animation \
  -c:a aac -b:a 192k -shortest STRAY.mp4
```

画面没有片名、字幕和片尾演职员表。署名在这里。

## 声音来源

曲子是本片原作，程序合成，不取自现成曲库。

现场声用的是录音，不是生成的音效：

- 引擎：Koenigsegg Regera 车内加速片段，Mr JWW，[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)
- 警笛：华盛顿街区录到的美国警车笛声，lezer，公有领域
- 旋翼：JoBrodie 录的支奴干飞越格林尼治，[CC0](https://creativecommons.org/publicdomain/zero/1.0/)
- 玻璃：Gravity Sound 的玻璃破碎，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 爆炸：2009 年日内瓦 Escalade，野战炮 “Falco” 的实弹，[CC BY-SA 2.0 FR](https://creativecommons.org/licenses/by-sa/2.0/fr/)
- 雨：ezwa / PDSounds，公有领域

中文招牌用的是文泉驿微米黑。
