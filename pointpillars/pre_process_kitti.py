import argparse
import os.path
import sys
CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR)

def create_data_info_pkl(data_root, data_type, prefix, label=True, db=False):
    sep = os.path.sep
    print(f"Processing {data_type} data..")
    ids_file = os.path.join(CUR, 'dataset', 'ImageSets', f'{data_type}.txt')
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readline()]

    split = 'training' if label else 'testing'

    kitti_infos_dict={}
    if db:







def main(args):
    data_root = args.data_root
    prefix = args.prefix

    ## 1. train: create data infomation pkl file && create reduced point clouds
    ##           && create database(points in gt bbox) for data aumentation
    kitti_train_infos_dict = create_data_info_pkl(data_root, "train", prefix, db=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()


