#!/usr/bin/env python
# coding: utf-8

# In[127]:


import pandas as pd
import glob
import numpy as np


# In[128]:


import pandas as pd
import glob
path = r'C:\Users\ahmad_z4ita07\Downloads\data-20220503T080359Z-001\data' # use your path
all_files = glob.glob(path + "/*_tracks.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)


# In[129]:


frame


# In[130]:


df = frame
df


# In[131]:


len(df['trackId'].value_counts())


# In[132]:


np.min(df['trackId']), np.max(df['trackId'])


# In[134]:


grouped_df = df.groupby('trackId')


# In[135]:


dv1_list = list(grouped_df.std()['xVelocity'])
dv1_list


# In[136]:


dv2_list = list(grouped_df.std()['lonAcceleration'])
dv2_list


# In[137]:


stds_list = np.array(dv1_list)
means_list = np.array(grouped_df.mean()['xVelocity'])


# In[74]:


dv3_list = (100 * stds_list) / means_list
dv3_list


# In[75]:


stds_list2 = np.array(dv2_list)
means_list2 = np.array(grouped_df.mean()['lonAcceleration'])


# In[76]:


dv4_list = (100 * stds_list2) / means_list2
dv4_list


# In[77]:


grouped_by_deacc = df[df['lonAcceleration'] < 0].groupby('trackId')


# In[78]:


stds_deacc = np.array(grouped_by_deacc.std()['lonAcceleration'])


# In[79]:


means_deacc = np.array(grouped_by_deacc.mean()['lonAcceleration'])


# In[80]:


dv5_list = (100 * stds_deacc) / means_deacc
dv5_list


# In[81]:


dv6_list = list(grouped_df.mad()['xVelocity'])
dv6_list


# In[82]:


dv7_list = list(grouped_df.mad()['lonAcceleration'])
dv7_list


# In[83]:


percentile_25 = []
percentile_75 = []

for i in range(len(grouped_df)):
    percentile_25.append(np.percentile(grouped_df.get_group(i)['xVelocity'], 25))
    percentile_75.append(np.percentile(grouped_df.get_group(i)['xVelocity'], 75))


# In[84]:


percentile_25 = np.array(percentile_25)
percentile_25


# In[85]:


percentile_75 = np.array(percentile_75)
percentile_75


# In[86]:


dv8_list = 100 * ((percentile_75 - percentile_25) / (percentile_75 + percentile_25))
dv8_list


# In[87]:


percentile_25_dlong = []
percentile_75_dlong = []

for i in range(len(grouped_df)):
    percentile_25_dlong.append(np.percentile(grouped_df.get_group(i)['lonAcceleration'], 25))
    percentile_75_dlong.append(np.percentile(grouped_df.get_group(i)['lonAcceleration'], 75))


# In[88]:


percentile_25_dlong = np.array(percentile_25_dlong)
percentile_75_dlong = np.array(percentile_75_dlong)


# In[89]:


dv9_list = 100 * ((percentile_75_dlong - percentile_25_dlong) / (percentile_75_dlong + percentile_25_dlong))
dv9_list


# In[90]:


percentile_25_deacc = []
percentile_75_deacc = []

for i in range(len(grouped_df)):
    percentile_25_deacc.append(np.percentile(grouped_by_deacc.get_group(i)['lonAcceleration'], 25))
    percentile_75_deacc.append(np.percentile(grouped_by_deacc.get_group(i)['lonAcceleration'], 75))


# In[91]:


percentile_25_deacc = np.array(percentile_25_deacc)
percentile_75_deacc = np.array(percentile_75_deacc)


# In[92]:


dv10_list = 100 * ((percentile_75_deacc - percentile_25_deacc) / (percentile_75_deacc + percentile_25_deacc))
dv10_list


# In[93]:


# std list for xVelocity
stds_list 


# In[94]:


# means list for xVelocity
means_list


# In[95]:


condition = 2 * stds_list + means_list
condition


# In[96]:


# var = true_val if condition else false_val


# In[97]:


dv11_list = []
for i in range(len(grouped_df)):
    mylist = list(grouped_df.get_group(i)['xVelocity'].apply(lambda x : x if x >= condition[i] else 0))
    dv11_list.append(100 * sum(mylist) / len(mylist))


# In[98]:


dv11_list


# In[99]:


df.head()


# In[100]:


pos_acc = df[df.lonAcceleration > 0]
pos_acc_gp = pos_acc.groupby('trackId')


# In[101]:


stds_pos_acc = pos_acc_gp.std()['lonAcceleration']
means_pos_acc = pos_acc_gp.mean()['lonAcceleration']

stds_pos_acc = np.array(stds_pos_acc)
means_pos_acc = np.array(means_pos_acc)


# In[102]:


# std list for lonAcceleration
stds_pos_acc


# In[103]:


# means list for lonAcceleration
means_pos_acc


# In[104]:


condition2 = 2 * stds_pos_acc + means_pos_acc
condition2


# In[105]:


dv12_list = []
for i in range(len(grouped_df)):
    mylist = list(pos_acc_gp.get_group(i)['lonAcceleration'].apply(lambda x : x if x >= condition2[i] else 0))
    dv12_list.append(100 * sum(mylist) / len(mylist))


# In[106]:


dv12_list


# In[107]:


neg_acc = df[df.lonAcceleration < 0]
neg_acc_gp = pos_acc.groupby('trackId')


# In[108]:


stds_neg_acc = neg_acc_gp.std()['lonAcceleration']
means_neg_acc = neg_acc_gp.mean()['lonAcceleration']

stds_neg_acc = np.array(stds_neg_acc)
means_neg_acc = np.array(means_neg_acc)


# In[109]:


condition3 = 2 * stds_neg_acc + means_neg_acc
condition3


# In[125]:


dv13_list = []
for i in range(len(grouped_df)):
    mylist = list(neg_acc_gp.get_group(i)['lonAcceleration'].apply(lambda x : x if x >= condition3[i] else 0))
    dv13_list.append(100 * sum(mylist) / len(mylist))


# In[126]:


dv13_list


# In[138]:


mydict = {}
for i in range(13):
    mystr = f"dv{i+1}_list"
    mydict[mystr] = eval(mystr)


# In[139]:


mydict


# In[142]:


df_new = pd.DataFrame(mydict)
df_new


# In[1]:


df_new.to_csv("ai_project.csv")

