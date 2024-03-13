import os
import shutil

def list_dirs(root=r'L:\data\kitti\odometry\dataset\sequences'):
    all_frames = []

    all_seq = os.listdir(root)
    all_seq.sort()
    
    for seq_name in all_seq:
        seq_path = os.path.join(root, seq_name)
        seq_path_velodyne = os.path.join(seq_path, 'velodyne')
        all_velo_frame_name = os.listdir(seq_path_velodyne)
        all_velo_frame_name.sort()
        seq_path_image_2 = os.path.join(seq_path, 'image_2')
        seq_path_calib = os.path.join(seq_path, 'calib.txt')

        for frame_name in all_velo_frame_name:
            frame={}
            frame['name']=frame_name[:6]
            frame['velo_path'] = os.path.join(seq_path_velodyne, frame_name)
            frame['im_path'] = os.path.join(seq_path_image_2, frame_name[:6]+'.png')
            frame['calib_path'] = os.path.join(seq_path_calib)
            all_frames.append(frame)
    return all_frames

def copy_calib(in_path, out_path):
    with open(in_path) as f:
        all_info = f.readlines()
        Tr = 'Tr_velo_to_cam'+all_info[4][2:]

        with open(out_path, 'w+') as f2:
            for i in range(4):
                f2.write(all_info[i])
            f2.write(Tr)


def move_data(input_path_dict, out_root='sampled_odometry', sampling=4):

    sampled_path = []
    for i in range(0, len(input_path_dict), sampling):
        sampled_path.append(input_path_dict[i])

    os.makedirs(out_root, exist_ok=True)

    for i, input_frame in enumerate( sampled_path):
        print(i,'/',len(sampled_path))
        name = str(i).zfill(6)
        in_velo_path = input_frame['velo_path']
        in_im_path = input_frame['im_path']
        in_calib_path = input_frame['calib_path']

        out_velo_root = os.path.join(out_root, 'velodyne')
        out_im_root = os.path.join(out_root, 'image_2')
        out_calib_root = os.path.join(out_root, 'calib')

        os.makedirs(out_velo_root, exist_ok=True)
        os.makedirs(out_im_root, exist_ok=True)
        os.makedirs(out_calib_root, exist_ok=True)

        out_velo_path = os.path.join(out_velo_root, name+'.bin')
        out_im_path = os.path.join(out_im_root, name+'.png')
        out_calib_path = os.path.join(out_calib_root, name+'.txt')

        shutil.copy(in_velo_path,out_velo_path)
        shutil.copy(in_im_path, out_im_path)
        copy_calib(in_calib_path, out_calib_path)

if __name__ == '__main__':
    import sys
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    input_path_dict = list_dirs(in_path)
    move_data(input_path_dict, out_path)
