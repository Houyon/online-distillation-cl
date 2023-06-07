import argparse
import subprocess
import time
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


if __name__ == "__main__":
    args = get_args()

    command = f'python listener.py --link {args.link} --output-dir {args.output_dir} --fps {args.fps}'
    timeout = 60

    p = subprocess.Popen(command, shell=True)
    file_to_check = f'{args.output_dir}/temp.mp4'

    while(True):
        try:
            latest_update_ms = os.path.getmtime(file_to_check)
            now_ms = time.time()
            if now_ms - latest_update_ms > timeout:
                p.kill()
                os.remove(file_to_check)
                p = subprocess.Popen(command, shell=True)
        except:
            continue