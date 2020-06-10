import pandas as pd


sienax_df = pd.read_csv('../data/SIENAX_FIRST_GM_parcellation_IDPs_forSam.txt', delim_whitespace=True)
print(sienax_df)
abcd_psb_df = pd.read_excel('../data/ABCD_PSB01.xlsx')
print(abcd_psb_df)
abcd_ysr_df = pd.read_excel('../data/ABCD_YSR01.xlsx')
print(abcd_ysr_df)
psb_df = pd.read_excel('../data/PSB01.xlsx')
print(psb_df)

im_dfs = [sienax_df]
non_im_dfs = [abcd_psb_df, abcd_ysr_df, psb_df]
all_dfs = im_dfs + non_im_dfs

for df in all_dfs:
    df.set_index('SUBJECTKEY', inplace=True, verify_integrity=True)
    print(df)
    print(df.dtypes)
    print(df.memory_usage(deep=True))
    print(df.memory_usage(deep=True).sum())
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'].astype('category')
    for column in ['EVENTNAME', 'SRC_SUBJECT_ID']:
        if column in df.columns:
            df[column] = df[column].astype('string')
    print(df.dtypes)
    print(df.memory_usage(deep=True))
    print(df.memory_usage(deep=True).sum())

print(sienax_df.join(abcd_ysr_df.add_suffix('_abcd_ysr').join(abcd_psb_df.add_suffix('_abcd_psb'), how='outer'), how='outer').columns)
# print(sienax_df.join(abcd_psb_df, how='outer').columns)
