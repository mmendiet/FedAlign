import os
import collections
import pandas as pd
import numpy as np
import re
import argparse

def main(log_dir):
    entries = collections.defaultdict(list)
    diags = collections.defaultdict(list)
    for folder in sorted(os.listdir(log_dir)):
        input_file_path = os.path.join(log_dir, folder, '{}.log'.format(folder))
        if os.path.isfile(input_file_path):
            inLog = open(input_file_path, 'r')
            lines = inLog.readlines()
            inLog.close()

            for line in lines:
                if '.pt' in line and '\t' in line:
                    method, dataset, run = line.split(':')[2].split('\t')
                    run = run.strip().split('/')[-1].split('.')[0]
                elif 'Eigen' in line:
                    eigenvalue = float(re.search('\[(\d*.\d*)', line).group(1))
                elif 'Trace' in line:
                    trace = float(line.split(' ')[1])
                elif 'Acc =' in line:
                    acc = line.split('= ')[1].split(' ')[0]
            entries[(method, dataset)].append({'run': run.strip(), 'eigen': eigenvalue, 'trace': trace, 'acc': acc})
            np_file_path = os.path.join(log_dir, folder, 'diag.npy')
            diag = np.load(np_file_path)
            diags[(method, dataset)].append((run.strip(), diag))
    for key in entries:
        diags_list = diags[key]
        norm_matrix = np.full( (len(diags_list)-1, len(diags_list)-1), 0.0)
        dir_matrix = np.full( (len(diags_list)-1, len(diags_list)-1), 0.0)
        for run, diag in diags_list:
            for run2, diag2 in diags_list:
                if 'server' not in run and 'server' not in run2:
                    run_num = int(run.split('.pt')[0].split('_')[-1])
                    run2_num = int(run2.split('.pt')[0].split('_')[-1])
                    norm_matrix[run_num][run2_num] = (np.linalg.norm(diag) - np.linalg.norm(diag2))**2
                    dir_matrix[run_num][run2_num] = np.dot(diag, diag2)/(np.linalg.norm(diag)*np.linalg.norm(diag2))
        print('{}: {}'.format('Norm', np.mean(norm_matrix)))
        print('{}: {}'.format('Direction', np.mean(dir_matrix)))

    for key in entries:
        df = pd.DataFrame(entries[key])
        df = df.sort_values('run')
        df.to_csv('{}/{}_data.csv'.format(log_dir, key[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, metavar='N',
                        help='Directory containing hessian results')
    args, unknown = parser.parse_known_args()
    main(args.log_dir)
