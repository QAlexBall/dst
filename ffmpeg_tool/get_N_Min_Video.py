import click
import os
import subprocess


def get_video_length(video_name):
    length = 0
    cmd = "ffmpeg -i {} 2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//".format(video_name)
    output = subprocess.check_output(cmd, shell=True).decode('utf-8')
    for i, time in enumerate(output.split(':')):
        time = int(float(time))
        if i == 0:
            length += time * 60 * 60
        if i == 1:
            length += time * 60
        if i == 2:
            length += time
    return length


@click.command()
@click.option("--n_min", help="get N min video")
@click.option("--video", help="source video path")
@click.option("--save_path", help="save cropped video path")
def get_N_Min_Video(n_min, video, save_path):
    """
    get N min video from source video
    """
    n_min = int(n_min)
    video_N_second = get_video_length(video)
    start_cut = 0
    if os.path.exists('./' + save_path):
        pass
    else:
        os.mkdir('./' + save_path)
    while video_N_second > start_cut:
        #  cmd = 'ffmpeg -y -ss 00:00:00 -t 00:23:00 -i 401-2.mp4 -vcodec copy -acodec copy -c:a aac 401-2video.mp4'
        cmd = 'ffmpeg -y -i {} -ss {} -t {} ./{}/{}~{}.mp4'.format(
            video,
            start_cut, n_min * 60,
            save_path,
            start_cut, (start_cut + n_min * 60))
        print(cmd)
        os.system(cmd)
        start_cut += n_min * 60

get_N_Min_Video()

#  if __name__ == '__main__':
    #  N = 20
    #  video_path = os.path.join(os.path.dirname(__file__), '301.mp4')
    #  get_N_Min_Video(N, video_path)
