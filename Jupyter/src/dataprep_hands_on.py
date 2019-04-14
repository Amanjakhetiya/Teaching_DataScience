import csv
with open('data/tab_delimited_stock_prices.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        # print("Date {}, Stock {} Price{}".format(date, symbol, closing_price))

with open('data/colon_delimited_stock_prices.txt', 'r') as f:
    reader = csv.DictReader(f, delimiter=':')
    for row in reader:
        date = row["date"]
        symbol = row["symbol"]
        closing_price = float(row["closing_price"])
        # print("Date {}, Stock {} Price {}".format(date, symbol, closing_price))

import pandas as pd
iris  = pd.read_csv('data/iris.csv')
print(iris.head())
# print(iris.describe())



# We'll also import seaborn, a Python graphing library
# import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
# warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
# iris.plot(kind="scatter", x="sepal_length_cm", y="sepal_width_cm")

# sns.FacetGrid(iris, hue="class", size=5) \
#    .map(plt.scatter, "sepal_length_cm", "sepal_width_cm") \
#    .add_legend()

# sns.boxplot(x="class", y="petal_length_cm", data=iris)

# sns.pairplot(iris, hue="class", size=3)
# plt.show()