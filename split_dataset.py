import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input')
args = parser.parse_args()



df = pd.read_csv(args.input, '\t')
df = df.drop_duplicates()



df = df.sample(frac=1)

split_index = int(len(df) * 0.1)
test_df  = df.iloc[0 : split_index]
train_df = df.iloc[split_index : ]

test_df.to_csv(args.input.split('.')[0] + '_test.tsv', '\t', index=False)
train_df.to_csv(args.input.split('.')[0] + '_train.tsv', '\t', index=False)