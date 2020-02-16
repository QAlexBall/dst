import os
import cv2
import sys
import json
import datetime
import asyncio
import numpy as np
from bs4 import BeautifulSoup
from collections.abc import Iterable

sys.path.append('..')
from slack_utils import send_slack_alert

DISTANCE = 0.75
X1_OFFSET = 10
X2_OFFSET = 10
Y1_OFFSET = 10
Y2_OFFSET = 10
MOVED_COUNT = 3
NONE_COUNT = 3
BUCKETS_FILE = './users_bucket.json'
MONTH_TO_STRING = {6: 'Jun', 7: 'Jul', 8: 'Aug'}


def send_to_slack(message, up=False):
    payload = {"text": " %s" % (message)}
    send_slack_alert(payload)


def download_current_hour_folder(bucket_name, stream):
    utc_time = datetime.datetime.utcnow()
    print(utc_time)
    stream_path = str(bucket_name) + '/' + str(stream)
    cmd = 'rm -rf ./{}/current && mkdir ./{}/current/ && aws s3 sync s3://{}/{}/{}/{}/{}/annotated ./{}/current/'.format(
        stream_path,
        stream_path,
        bucket_name,
        stream,
        MONTH_TO_STRING[utc_time.month],
        utc_time.day,
        utc_time.hour - 1,
        stream_path
    )
    try:
        if not os.path.exists('./{}'.format(bucket_name)):
            os.system('mkdir ./{}'.format(bucket_name))
        if not os.path.exists('./{}'.format(stream_path)):
            os.system('mkdir ./{}'.format(stream_path))
        os.system(cmd)
        return True
    except Exception:
        print('can not download this folder')


def find_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kpts, descs = sift.detectAndCompute(gray, None)
    return image, kpts, descs


def match_image(img1, kpts1, descs1, img2, kpts2, descs2):
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
    matches = matcher.knnMatch(descs1, descs2, 2)
    matches = sorted(matches, key=lambda x: x[0].distance)
    good = [m1 for (m1, m2) in matches if m1.distance < DISTANCE * m2.distance]
    canvas = img2.copy()
    dst = None
    if len(good) > 1:
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if isinstance(M, Iterable):
            h, w = img1.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            # cv2.pipelines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
            # matched = cv2.drawMatches(img1, kpts1, canvas, kpts2, good, None)#,**draw_params)
            # perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
            # found = cv2.warpPerspective(img2,perspectiveM,(w,h))
            # cv2.imshow('matched', matched);
            # cv2.imshow('found', found);
            # cv2.waitKey(1000);
            # cv2.destroyAllWindows()
    else:
        print("can't find enough feature point")
    return dst


def find_offset_with_current_folder(bucket_name, stream, img1, kpts1, descs1, x1, x2, y1, y2):
    camera_state = "notmove"
    path = './{}/{}/current/'.format(bucket_name, stream)
    images = os.listdir(path)
    camera_state = 'notmove'
    may_moved_count = 0
    dst_none_count = 0
    for im in images:
        if im != 'annotated':
            print(path + im)
            image = cv2.imread(path + im)
            img2, kpts2, descs2 = find_feature(image)
            dst = match_image(img1, kpts1, descs1, img2, kpts2, descs2)
            if dst is not None:
                print('==> bucket_name: ' + bucket_name + '\n==> stream: ' + stream)
                print('x: ' + str(y1 - dst[0][0][1]) + ':' + str(y2 - dst[2][0][1]))
                print('y: ' + str(x1 - dst[0][0][0]) + ':' + str(x2 - dst[2][0][0]))
                if abs(x1 - dst[0][0][0]) > X1_OFFSET \
                        or abs(y1 - dst[0][0][1]) > Y1_OFFSET \
                        or abs(x2 - dst[2][0][0]) > X2_OFFSET \
                        or abs(y2 - dst[2][0][1]) > Y2_OFFSET:
                    print('==> * moved * <==')
                    may_moved_count += 1
                else:
                    may_moved_count = 0
                    print('==> * not move * <==')
            else:
                dst_none_count += 1
            print("===> may_moved_count ", may_moved_count)
            print("===> dst_none_count", dst_none_count)
            print('\n')
    if may_moved_count >= MOVED_COUNT or dst_none_count >= NONE_COUNT:
        camera_state = 'moved'
    return camera_state


def get_first_image_feature(bucket, stream, roi, moved=False):
    bucket_name = bucket.get('bucket_name', None)
    stream_name = stream.get('stream', None)

    first_image = './{}/{}/current_image.jpg'.format(bucket_name, stream_name)
    if moved:
        # label image when start or camera moved. update roi
        roi = check_roi(bucket, stream, moved=True)
    x1, y1, x2, y2 = roi[0], roi[1], roi[2], roi[3]
    first_im = cv2.imread(first_image)
    crop_img = first_im[y1:y2, x1:x2]
    img, kpts, descs = find_feature(crop_img)
    cv2.imwrite('./{}/{}/first_cropped_image.jpg'.format(bucket_name, stream_name), crop_img)
    return img, kpts, descs, x1, x2, y1, y2


def label_image(bucket_name, stream_name, first_image):
    # label image when start or camera moved.
    os.system('labelImg {}'.format(first_image))
    os.system('mv ./current_image.xml ./{}/{}/'.format(bucket_name, stream_name))

    first_image_xml = './{}/{}/current_image.xml'.format(bucket_name, stream_name)
    soup = BeautifulSoup(open(first_image_xml), 'xml')
    x1, x2 = int(soup.xmin.string), int(soup.xmax.string)
    y1, y2 = int(soup.ymin.string), int(soup.ymax.string)
    new_roi = [x1, y1, x2, y2]
    return new_roi


def update_roi(current_bucket_name, current_stream_name, new_roi):
    f = open(BUCKETS_FILE, 'r')
    buckets = json.load(f)
    f.close()
    for bucket_position, bucket in enumerate(buckets):
        if bucket.get('bucket_name', None) == current_bucket_name:
            for stream_position, stream in enumerate(bucket.get('streams', None)):
                if stream.get('stream', None) == current_stream_name:
                    buckets[bucket_position]['streams'][stream_position]['roi'] = new_roi
    f = open(BUCKETS_FILE, 'w')
    json.dump(buckets, f, indent=2)
    f.close()


def check_roi(bucket, stream, moved=False):
    current_bucket_name = bucket.get('bucket_name', None)
    current_stream_name = stream.get('stream', None)
    roi = stream.get('roi', None)
    print("==> check roi: ", current_bucket_name, current_stream_name)
    first_image = './{}/{}/current_image.jpg'.format(current_bucket_name, current_stream_name)
    if not os.path.exists('./{}/{}/current'.format(current_bucket_name, current_stream_name)) or moved:
        download_current_hour_folder(current_bucket_name, current_stream_name)
    # generate first image
    for image in os.listdir('./{}/{}/current/'.format(current_bucket_name, current_stream_name)):
        os.system('cp ./{}/{}/current/{} {}'.format(current_bucket_name, current_stream_name, image, first_image))
    if stream.get('roi', None) == [] or moved:
        print(current_bucket_name, current_stream_name, "download_current_hour_folder done!")
        roi = label_image(current_bucket_name, current_stream_name, first_image)
        update_roi(current_bucket_name, current_stream_name, roi)
    return roi


def check_all_roi():
    f = open(BUCKETS_FILE, 'r')
    buckets = json.load(f)
    f.close()
    for bucket in buckets:
        for stream in bucket.get('streams', None):
            roi = check_roi(bucket, stream)
            print("==> roi: ", roi)


async def start_monitor(bucket, stream):
    bucket_name = bucket.get('bucket_name', None)
    hour = stream.get('hour', None)
    roi = stream.get('roi', None)
    stream_name = stream.get('stream', None)
    sleep_time = 1800
    if hour:
        sleep_time = int(float(hour) * 60 * 60)
    img1, kpts1, descs1, x1, x2, y1, y2 = get_first_image_feature(bucket, stream, roi)
    while True:
        print("\n==> start download ", bucket_name, stream_name)
        success = download_current_hour_folder(bucket_name, stream_name)
        if success:
            camera_state = find_offset_with_current_folder(bucket_name, stream_name, img1, kpts1, descs1, x1, x2, y1,
                                                           y2)
            while camera_state == 'moved':
                print("==> camera stream " + stream_name + " moved")
                # send_to_slack(stream + ' camera moved')
                # if moved,it need to re-select position and re-calculate offset
                download_current_hour_folder(bucket_name, stream_name)
                # img1, kpts1, descs1, x1, x2, y1, y2 = get_first_image_feature(bucket, stream, roi, moved=True)
                camera_state = find_offset_with_current_folder(bucket_name, stream_name, img1, kpts1, descs1, x1, x2,
                                                               y1, y2)

        print('wait {} hour...'.format(sleep_time / 60 / 60))
        await asyncio.sleep(sleep_time)


def main():
    f = open(BUCKETS_FILE, 'r')
    buckets = json.load(f)
    f.close()
    check_all_roi()
    ioloop = asyncio.get_event_loop()
    tasks = []
    for bucket in buckets:
        for stream in bucket.get('streams', None):
            tasks.append(ioloop.create_task(start_monitor(bucket, stream)))
    ioloop.run_until_complete(asyncio.wait(tasks))
    ioloop.close()


if __name__ == '__main__':
    main()

# use argparse run with one camera
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Monitor camera moved!")
#     parser.add_argument('-b', '--bucket_name', required=True)
#     parser.add_argument('-s', '--stream', required=True)
#     parser.add_argument('-hour', '--hour', help='wait time')
#     args = parser.parse_args()

#     bucket_name = args.bucket_name
#     stream = args.stream
#     sleep_time = 1800
#     if args.hour:
#         sleep_time = int(float(args.hour) * 60 * 60)

#     download_current_hour_folder(bucket_name, stream)
#     print("download_current_hour_folder done!")
#     img1, kpts1, descs1, x1, x2, y1, y2 = get_first_image_feature(bucket_name, stream)
#     while True:
#         success = download_current_hour_folder(bucket_name, stream)
#         if success:
#             camera_state = find_offset_with_current_folder(bucket_name, stream, img1, kpts1, descs1, x1, x2, y1, y2)
#             if camera_state == 'moved':
#                 send_to_slack(stream + ' camera moved')
#         print('wait {} hour...'.format(sleep_time / 60 / 60))
#         time.sleep(sleep_time) 


# find offset with current image def find_offset_with_current_image(bucket_name, stream, img1, kpts1, descs1, x1, x2,
# y1, y2): camera_state = "notmove" image = cv2.imread('./{}/{}/current_image.jpg'.format(bucket_name, stream)) img2,
# kpts2, descs2 = find_feature(image) dst = match_image(img1,kpts1, descs1, img2, kpts2, descs2) if dst is not None:
# print('x: ' + str(y1 - dst[0][0][1]) + ':' + str(y2 - dst[2][0][1])) print('y: ' + str(x1 - dst[0][0][0]) + ':' +
# str(x2 - dst[2][0][0])) may_moved_count = 0 if abs(x1 - dst[0][0][0]) > 10 or abs(y1 - dst[0][0][1]) > 10 or abs(x2
# - dst[2][0][0]) > 10 or abs(y2 - dst[2][0][1]) > 10: print('==> * moved * <==') may_moved_count += 1 # if may moved
# count > a number, defined camera_state moved if may_moved_count >= 1: camera_state = 'moved' else: may_moved_count
# = 0 print('==> * not move * <==') print('\n') return camera_state
