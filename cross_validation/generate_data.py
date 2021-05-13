import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--undirected', default=0, type=int)
    parser.add_argument('--diag_one', default=0, type=int)
    parser.add_argument('--normalized', default=0, type=int)
    opt = parser.parse_args()

    suffix = ""

    process("./datasets/ReHol/train_triplets.txt", opt, suffix)
    process("./datasets/ReHol/dev_triplets.txt", opt, suffix)
    process("./datasets/ReHol/test_triplets.txt", opt, suffix)
