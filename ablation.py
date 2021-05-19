import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=['rest14', 'rest15', 'rest16', 'lap14', 'reli', 'rehol'])
    parser.add_argument("--models", nargs="+", default=['bote', 'ote', 'cmla'])
    opt = parser.parse_args()