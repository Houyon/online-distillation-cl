from glob import glob
from tqdm import tqdm
import cv2

def concatenate_train_eval(
    train_path: str,
    eval_path: str, 
    train_fps: int, 
    eval_fps: int,
    output_path: str,
):
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), train_fps, (1920, 1080))
    
    read_every = eval_fps // train_fps
    read_count = 0
    
    eval_video = cv2.VideoCapture(eval_path)
    while(True):
        r, frame = eval_video.read()
        if not r:
            break
        
        if read_count % read_every == 0:
            output_video.write(frame)
        
        read_count += 1
    
    train_video = cv2.VideoCapture(train_path)
    while(True):
        r, frame = train_video.read()
        if not r:
            break
            
        output_video.write(frame)
            
    output_video.release()
    
    
if __name__ == '__main__':
    
    from arguments import args
    
    input_dir = os.path.join(args.dataset, 'videos')
    output_dir = os.path.join(args.dataset, 'concatenate')
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = glob(os.path.join(input_dir, '*.mp4'))
    
    # Should make some verifications whether the folder contains correct videos
    
    total_digits = len(video_files)/2
    for i in tqdm(range(0, len(video_files), 2)):
        eval_path, train_path = video_files[i], video_files[i+1]

        used_digits = len(str(i//2))
        output_path =  f'{output_dir}/{str(0)*(total_digits-used_digits)}{i//2}.mp4'

        concatenate_train_eval(train_path, eval_path, args.trainfps, args.evalfps, output_path)