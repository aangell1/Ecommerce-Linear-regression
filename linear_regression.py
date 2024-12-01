import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./ecommerce.csv")
df.head()
df.info()
df.describe()

#EDA
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)
sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.4})
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha': 0.3})

from sklearn.model_selection import train_test_split

X = df[['Avg. Session Length', 'Time on App', 'Time on Webstie', 'Length of Membership']]
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

#training the model

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_
cdf = pd.DataFrame(lm.coef_, X.columns, columns={'Coef'})
print(cdf)

#predictions

predictions = lm.predict(X_test)
print(predictions)
sns.scatterplot(predictions, y_test)
plt.xlabel("predictions")
plt.title("evaluation of LM model")
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

print("Mean Absolute Error: ", mean_absolute_error(ytest, predictions))
print("Mean Squared Error: ", mean_sqaured_error(y_test, predictions))
print("RMSE: ", math.sqr(mean_squared_error(y_test, predictions)))

# residuals 

residuals = y_test - predictions
sns.displot(residuals)
sns.displot(residuals, bins=30, kde=True)

import pylab
import scipy.stats as stats

stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()






