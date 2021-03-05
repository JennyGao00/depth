import shutil
import os


def move_file(src_path, dst_path, file):
    # print('OK')
    # cmd = 'chmod -R +x ' + src_path
    # os.popen(cmd)
    f_src = os.path.join(src_path, file)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    f_dst = os.path.join(dst_path, file)
    shutil.move(f_src, f_dst)
    print('moved file: ' + str(f_src) + ' to ' + str(f_dst))


if __name__ == '__main__':
    src_path = '/media/gao/Gao106/NYUV2/data/NYU_normal/data1/depth1'
    dst_path = '/media/gao/Gao106/NYUV2/data/NYU_normal/data1/normal1'
    # files_name = os.listdir(src_path)
    files = [d for d in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, d))]

    # separate the depth.png from colors.png
    for file in files:
        if file[6:12] == 'normal':
            move_file(src_path, dst_path, file)

