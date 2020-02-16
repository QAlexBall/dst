# use ffmpeg concat videos with fileslist.txt

```shell
$ # use python to write all videos to fileslist.txt
$ python write_to_filelist.py
$ ffmpeg -y -f concat -i filelist.txt -c copy collected_301_20191125.mp4
```