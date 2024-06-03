#%%
%load_ext kedro.ipython
# %%
raw_dataset = context.catalog.load("raw_medical")
# create a dataset from a pandas dataframe
# %%
raw_dataset.head(2)
