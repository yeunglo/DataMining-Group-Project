import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/Users/jamieliu/Downloads/toxic.csv')
train.tail(10)
# nrow_train = train.shape[0]
x0 = train.iloc[:, 2:].sum()
# marking comments without any tags as "clean"
rowsums = train.iloc[:, 2:].sum(axis=1)
train['clean'] = (rowsums == 0)
# count number of clean entries
train['clean'].sum()
# plot
x1 = train.iloc[:, 2:].sum()
plt.figure(figsize=(8, 4))
ax = sns.barplot(x1.index, x1.values, alpha=0.8, palette="Blues_d")
plt.title("the count of comments per class")
plt.ylabel('Count of Comments', fontsize=12)
plt.xlabel('class', fontsize=12)
# adding the text labels
rects = ax.patches
labels = x1.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2, height+5, label, ha='center', va='bottom')

plt.show()

x2 = rowsums.value_counts()

# plot
plt.figure(figsize=(8, 4))
ax = sns.barplot(x2.index, x2.values, alpha=0.8, palette='GnBu_d')
plt.title("Multiple labels per comment")
plt.ylabel('Count of Comments', fontsize=12)
plt.xlabel('Count of labels', fontsize=12)

# adding the text labels
rectss = ax.patches
labels = x2.values
for rect, label in zip(rectss, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()

temp_df = train.iloc[:, 2:-1]
# filter temp by removing clean comments
# temp_df=temp_df[~train.clean]

corr = temp_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap='YlGnBu')

plt.show()



