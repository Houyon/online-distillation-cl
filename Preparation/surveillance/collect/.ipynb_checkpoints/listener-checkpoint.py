from vidgear.gears import CamGear
import cv2
import argparse
import time
import datetime
import os

def get_args():
    parser = argparse.ArgumentParser()

    # link to the video stream
    parser.add_argument('--link', type=str, default='https://www.youtube.com/watch?v=DjdUEyjx8GM')

    # directory for the writen videos
    parser.add_argument('--output-dir', default='videos/tokyo', type=str)

    # FPS
    parser.add_argument('--fps', default=1, type=int)

    return parser.parse_args()


def launch_stream(source):
    while(True):
        try:
            stream = CamGear(source=source, stream_mode=True, logging=True).start()
            return stream
        except:
            continue


def video(
    video_stream: str,
    directory: str,
    video_name: int, 
    temp_video_name: str,
    length: int, 
    fps: int, 
):
    
    # number of frames read so far for the current second
    frame_count = 1

    # number of seconds recorded
    current_length = 0

    stream = launch_stream(video_stream)

    if fps == -1:
        fps = stream.framerate

    read_ratio = stream.framerate // fps

    video = cv2.VideoWriter(f'{directory}/{temp_video_name}.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), stream.framerate, (1920, 1080))

    while(current_length < length):
        frame = stream.read()
        if frame is None:
            stream.stop()
            stream = launch_stream(video_stream)
            continue
        
        if frame_count % read_ratio == 0:
            # write frame into the current video
            video.write(frame)
            
            # if frame_count is equal to stream.framerate, we have written one second of our video
            if frame_count >= stream.framerate:
                current_length += 1
                # set back frame_count to 0
                frame_count = 0

        # increment the number of frames read so far for the current second
        frame_count += 1
    
    video.release()
    stream.stop()
    date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S')
    os.rename(f'{directory}/{temp_video_name}.mp4', f'{directory}/{date_now} {video_name}.mp4')


def write_videos(
    video_stream: str, 
    output_dir: str,
    fps: int,
):      
    while(True):
        #First write a video at full FPS
        video(
            video_stream=video_stream, 
            directory=output_dir,
            video_name='eval', 
            length=60,
            fps=-1,
            temp_video_name='temp'
        )

        #Rest of the video at 1 FPS
        video(
            video_stream=video_stream, 
            directory=output_dir,
            video_name='train', 
            length=3540,
            fps=fps,
            temp_video_name='temp'
        )

if __name__ == '__main__':
    args = get_args()

    video_stream = args.link
    output_dir = args.output_dir
    fps = args.fps

    print(f'Fetching video stream on {video_stream}')

    # start writing videos
    write_videos(
        video_stream=video_stream,
        output_dir=output_dir,
        fps=fps,
    )
