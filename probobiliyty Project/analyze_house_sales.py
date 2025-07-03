import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import os
import sys

# Auto-install missing packages
try:
    import openpyxl
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openpyxl'])
    import openpyxl

# --- Load Data ---
file_path = 'us_house_Sales_data.csv'
df = pd.read_csv(file_path)

# --- Data Cleaning ---
def clean_price(x):
    return float(str(x).replace('$','').replace(',',''))
def clean_area(x):
    return float(str(x).replace('sqft','').replace(',','').strip())
def clean_beds(x):
    return float(str(x).replace('bds','').replace('bd','').strip())
def clean_baths(x):
    return float(str(x).replace('ba','').strip())

df['Price'] = df['Price'].apply(clean_price)
df['Area (Sqft)'] = df['Area (Sqft)'].apply(clean_area)
df['Bedrooms'] = df['Bedrooms'].apply(clean_beds)
df['Bathrooms'] = df['Bathrooms'].apply(clean_baths)

# --- Sample 300 rows ---
sample = df.sample(n=300, random_state=42).reset_index(drop=True)

# --- Descriptive Statistics ---
desc_stats = sample[['Price','Area (Sqft)','Bedrooms','Bathrooms','Days on Market']].describe().T

# --- Normal Distribution & PDF Plots ---
plt.figure(figsize=(8,4))
sns.histplot(sample['Price'], kde=True, bins=30)
plt.title('Histogram & PDF of House Price')
plt.xlabel('Price')
plt.tight_layout()
plt.savefig('price_hist.png')
plt.close()

plt.figure(figsize=(8,4))
stats.probplot(sample['Price'], dist="norm", plot=plt)
plt.title('Q-Q Plot of House Price')
plt.tight_layout()
plt.savefig('price_qq.png')
plt.close()

# --- Z-scores & Probability Example ---
sample['Price_zscore'] = (sample['Price'] - sample['Price'].mean()) / sample['Price'].std()
# Example: Probability Price > $1,000,000
z = (1_000_000 - sample['Price'].mean()) / sample['Price'].std()
prob_gt_1m = 1 - stats.norm.cdf(z)

# --- Confidence Intervals ---
mean = sample['Price'].mean()
std = sample['Price'].std()
n = len(sample)
conf_int_z = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(n))
conf_int_t = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n))

# --- Hypothesis Testing ---
# H0: mean <= 500,000, H1: mean > 500,000
mu0 = 500_000
z_stat = (mean - mu0) / (std/np.sqrt(n))
p_value_z = 1 - stats.norm.cdf(z_stat)
t_stat, p_value_t = stats.ttest_1samp(sample['Price'], mu0)
p_value_t = p_value_t/2 if mean > mu0 else 1-p_value_t/2

# --- Regression & Correlation ---
X = sample[['Area (Sqft)']]
y = sample['Price']
reg = LinearRegression().fit(X, y)
sample['Predicted Price'] = reg.predict(X)
cov = np.cov(sample['Area (Sqft)'], sample['Price'])[0,1]
pearson_corr = np.corrcoef(sample['Area (Sqft)'], sample['Price'])[0,1]
r2 = reg.score(X, y)

plt.figure(figsize=(8,5))
sns.scatterplot(x='Area (Sqft)', y='Price', data=sample)
plt.plot(sample['Area (Sqft)'], sample['Predicted Price'], color='red', label='Regression Line')
plt.title('Area vs. Price with Regression Line')
plt.legend()
plt.tight_layout()
plt.savefig('regression_plot.png')
plt.close()

# --- Write to Excel ---
with pd.ExcelWriter('house_sales_analysis.xlsx', engine='openpyxl') as writer:
    sample.to_excel(writer, sheet_name='Sample Data', index=False)
    desc_stats.to_excel(writer, sheet_name='Descriptive Stats')
    # Z-score and probability
    pd.DataFrame({
        'Z for $1,000,000':[z],
        'P(Price > $1,000,000)':[prob_gt_1m]
    }).to_excel(writer, sheet_name='Z-Score & Prob', index=False)
    # Confidence intervals
    pd.DataFrame({
        'Mean':[mean],
        'Std':[std],
        '95% CI (Z)':[conf_int_z],
        '95% CI (t)':[conf_int_t],
        'Error Bound (Z)':[conf_int_z[1]-mean],
        'Error Bound (t)':[conf_int_t[1]-mean]
    }).to_excel(writer, sheet_name='Confidence Intervals', index=False)
    # Hypothesis test
    pd.DataFrame({
        'Z-stat':[z_stat],
        'Z p-value':[p_value_z],
        't-stat':[t_stat],
        't p-value':[p_value_t],
        'H0: mean <= 500,000':[p_value_z < 0.05 or p_value_t < 0.05]
    }).to_excel(writer, sheet_name='Hypothesis Test', index=False)
    # Regression & Correlation
    pd.DataFrame({
        'Covariance':[cov],
        'Pearson r':[pearson_corr],
        'R^2':[r2],
        'Intercept':[reg.intercept_],
        'Slope':[reg.coef_[0]]
    }).to_excel(writer, sheet_name='Regression', index=False)

# --- Add images to Excel (optional, manual step) ---
print('Analysis complete!')
print('Check house_sales_analysis.xlsx and the PNG files for plots. You can insert the PNGs into the Excel file if desired.') 