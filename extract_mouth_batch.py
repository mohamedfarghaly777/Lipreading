import sys
sys.path.append('../../')
from videos import Video
import os, fnmatch, errno
import time
from skimage import io
#import threading

"""
class MyThread(threading.Thread):
    def __init__(self, filepath, FACE_PREDICTOR_PATH, targetdir1):
        threading.Thread.__init__(self)
        self.filepath = filepath
        self.FACE_PREDICTOR_PATH = FACE_PREDICTOR_PATH
        self.targetdir1 = targetdir1
    def run(self):
        print("Processing: {}".format(self.targetdir1))
        self.video = Video(vtype='face', face_predictor_path=self.FACE_PREDICTOR_PATH).from_video(self.filepath)
        self.mkdir_p(self.targetdir1)
        i = 0
        for frame in self.video.mouth:
            io.imsave(os.path.join(self.targetdir1, "mouth_{0:03d}.png".format(i)), frame)
            i += 1

    def mkdir_p(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise
"""
SOURCE_PATH = sys.argv[1]
SOURCE_EXTS = sys.argv[2]
TARGET_PATH1 = sys.argv[3]
TARGET_PATH2 = sys.argv[4]
FACE_PREDICTOR_PATH = sys.argv[5]

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename
                
def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise
            
threads = []
start = time.time()
cnt = 1
print(TARGET_PATH1)
print(TARGET_PATH2)
for filepath in find_files(SOURCE_PATH, SOURCE_EXTS):
    endpath = os.path.splitext(filepath)[0].split('/')[-1]
    targetdir1 = os.path.join(TARGET_PATH1, endpath)
    targetdir2 = os.path.join(TARGET_PATH2, endpath)
    if cnt > 400:
        targetdir1 = targetdir2
    print("Processing: {}".format(targetdir1))
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH).from_video(filepath)
    mkdir_p(targetdir1)
    i = 0
    for frame in video.mouth:
        io.imsave(os.path.join(targetdir1, "mouth_{0:03d}.png".format(i)), frame)
        i += 1
    cnt += 1
    """
    threads.append(MyThread(filepath, FACE_PREDICTOR_PATH, targetdir1))
    threads[cnt - 1].start()
    if (cnt % 2) == 0:
        for i in range(cnt - 2, len(threads)):
            threads[i].join()
    """    
end = time.time()
print("Time is " + str(end - start))