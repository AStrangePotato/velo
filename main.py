
from pose_helper import track_pose
from audio_analyzer import loudest_sound_time

if __name__ == '__main__':
    c = track_pose('input_video.mp4')
    print(c)
    s = loudest_sound_time("input_video.mp4")
    print(s)