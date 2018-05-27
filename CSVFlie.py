import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import re
pd.set_option('display.max_colwidth',1000)
TAGS = set(['sd', 'b', 'sv', 'aa', '%', 'ba', 'qy', 'x', 'ny', 'fc', 'qw', 'nn', 'bk', 'h', 'qy^d', 'fo_o_fw_by_bc', 'bh', '^q', 'bf', 'na', 'ad', '^2', 'b^m', 'qo', 'qh', '^h', 'ar', 'ng', 'br', 'no', 'fp', 'qrr', 'arp_nd', 't3', 'oo_co_cc', 't1', 'bd', 'aap_am', '^g', 'qw^d', 'fa','ft' ])



filename = "swda_stats.csv"
data = pd.read_csv(filename)

data = data[data.utts_length >1]


data['utterances'] = data['utterances'].apply(lambda x: re.sub(r"[-',.;!:?%$#*&+=\")(_~/]", '', x))
data['utterances'] = data['utterances'].apply(lambda x: x.lower())
data['utterances'] = data['utterances'].apply(lambda x: x.split())
data['utts_length'] = data['utterances'].apply(lambda x: len(x))
data = data.replace("+", 'x')
data = data.replace('fo_o_fw_"_by_bc', 'fo_o_fw_by_bc')

length = data['utts_length'].value_counts()

print(length)
print(data.utts_length.unique())

u10 = len(data[data.utts_length <= 10])
print("u10 = ",u10)
u20 = len(data[(data.utts_length > 10) & (data.utts_length <= 20)])
print("u20 = ",u20)
u30 = len(data[(data.utts_length > 20) & (data.utts_length <= 30)])
print("u30 = ",u30)
u40 = len(data[(data.utts_length > 30) & (data.utts_length <= 40)])
print("u40 = ",u40)
u50 = len(data[(data.utts_length > 40) & (data.utts_length <= 50)])
print("u50 = ",u50)
u60 = len(data[(data.utts_length > 50) & (data.utts_length <= 60)])
print("u60 = ",u60)
u70 = len(data[(data.utts_length > 60) & (data.utts_length <= 70)])
print("u70 = ",u70)
u80 = len(data[(data.utts_length > 70) & (data.utts_length <= 80)])
print("u80 = ",u80)



exit()
temp = data[['utterances', 'tags']]


tag_series = data.tags.value_counts()


lt100 = tag_series[tag_series < 100].index # del
gt100_lt200 = tag_series[(200 > tag_series)  & (tag_series > 100)].index # x 80
gt200_lt250 = tag_series[(250 > tag_series)  & (tag_series > 200)].index # x 50
gt250_lt300 = tag_series[(300 > tag_series)  & (tag_series > 250)].index # x 35
gt300_lt500 = tag_series[(500 > tag_series)  & (tag_series > 300)].index # x 20
gt500_lt600 = tag_series[(600 > tag_series)  & (tag_series > 500)].index # x 18
gt600_lt700 = tag_series[(700 > tag_series)  & (tag_series > 600)].index # x 15
gt700_lt800 = tag_series[(800 > tag_series)  & (tag_series > 700)].index # x 14
gt800_lt900 = tag_series[(900 > tag_series)  & (tag_series > 800)].index # x 13
gt900_lt1000 = tag_series[(1000 > tag_series)  & (tag_series > 900)].index # x 12
gt1000_lt2000 = tag_series[(2000 > tag_series)  & (tag_series > 1000)].index # x7
gt2000_lt4000 = tag_series[(4000 > tag_series)  & (tag_series > 2000)].index # x6
gt4000_lt10000 = tag_series[(10000 > tag_series)  & (tag_series > 4000)].index # x3

temp = temp.loc[~temp.tags.isin(lt100)]

def over_sampling(df, factor):
    x = np.tile(df['utterances'], [factor])
    y = np.tile(df['tags'], [factor])
    sample = pd.DataFrame({'utterances': x, 'tags': y})
    return sample

# Over Sampling 1
df1 = temp.loc[temp.tags.isin(gt100_lt200)]
sample_1 = over_sampling(df1, 80)


# Over Sampling 2
df2 = temp.loc[temp.tags.isin(gt200_lt250)]
sample_2 = over_sampling(df2, 50)


# Over Sampling 3
df3 = temp.loc[temp.tags.isin(gt250_lt300)]
sample_3 = over_sampling(df3, 35)


# Over Sampling 4
df4 = temp.loc[temp.tags.isin(gt300_lt500)]
sample_4 = over_sampling(df4, 20)


# Over Sampling 5
df5 = temp.loc[temp.tags.isin(gt500_lt600)]
sample_5 = over_sampling(df5, 18)


# Over Sampling 6
df6 = temp.loc[temp.tags.isin(gt600_lt700)]
sample_6 = over_sampling(df6, 15)


# Over Sampling 7
df7 = temp.loc[temp.tags.isin(gt700_lt800)]
sample_7 = over_sampling(df7, 14)


# Over Sampling 8
df8 = temp.loc[temp.tags.isin(gt800_lt900)]
sample_8 = over_sampling(df8, 13)


# Over Sampling 9
df9 = temp.loc[temp.tags.isin(gt900_lt1000)]
sample_9 = over_sampling(df9, 12)


# Over Sampling 10
df10 = temp.loc[temp.tags.isin(gt1000_lt2000)]
sample_10 = over_sampling(df10, 7)


# Over Sampling 11
df11 = temp.loc[temp.tags.isin(gt2000_lt4000)]
sample_11 = over_sampling(df11, 6)


# Over Sampling 12
df12 = temp.loc[temp.tags.isin(gt4000_lt10000)]
sample_12 = over_sampling(df12, 3)



AAA = pd.concat([temp, sample_1, sample_2, sample_3,sample_4, sample_5, sample_6,sample_7, sample_8, sample_9, sample_10, sample_11, sample_12], axis = 0)

print(AAA.shape)
print(AAA.head())
AAA.to_csv("over_sampling.csv", index = False)
data.to_csv("swda_clip.csv", index = False)