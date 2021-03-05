import os

if __name__ == "__main__":
    src_path = "/media/gao/Gao106/NYUV2/data/NYU_normal/test/depth"
    outpath = "/home/gao/space/depth/data/test.txt"
    depthlist = ['depth/' + str(d) for d in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, d))]
    # print(depthlist)
    with open(outpath, 'w') as df:
        for depth in depthlist:
            df.write(depth + '\n')
    # print(os.getcwd())
    # depth = [line.rstrip() for line in open(os.path.join(src_path, 'train.txt'))]
    # print(depth)
