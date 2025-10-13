# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Importing Data
df = pd.read_csv('your_pile_path/Sample_Data_Seoul_SocioCrime_Data.csv')

# Reading columns
required_cols = ['District', 'CrimeRate', 'ReligionRatio', 'FiscalIndependence', 'VolunteerRate']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f'Missing columns in CSV: {missing}')

# Columns str to numeric
for col in ['CrimeRate', 'ReligionRatio', 'FiscalIndependence', 'VolunteerRate']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# NaN Value Drop
before = len(df)
df = df.dropna(subset=['CrimeRate','ReligionRatio','FiscalIndependence','VolunteerRate']).reset_index(drop=True)
after = len(df)
print(f'Loaded rows: {after} (dropped {before-after} rows with NaN)')

print('\n[Head]')
print(df.head().to_string(index=False))

print('\n[Describe]')
print(df[['CrimeRate','ReligionRatio','FiscalIndependence','VolunteerRate']].describe().to_string())

# Define Slope, Intercept, R^2, Linear Regression, Legends
def plot_scatter_with_fit(x, y, x_label, y_label='Crime Rate (per 1,000)', annotate=True, savepath=None):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Slope&Intercept
    slope, intercept = np.polyfit(x, y, 1)
    # R&R^2
    if x.std() == 0 or y.std() == 0:
        r2 = np.nan
    else:
        r = np.corrcoef(x, y)[0,1]
        r2 = r**2

    # Linear Regression
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    # Graph Plot
    plt.figure()
    plt.scatter(x, y, s=36, alpha=0.8)
    plt.plot(x_line, y_line, linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')

    # Legends
    txt = f'y = {intercept:.2f} + {slope:.3f}·x\nR² = {r2:.3f}'
    plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                   ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    if annotate and 'District' in df.columns:
        for i in range(len(df)):
            plt.annotate(str(df.loc[i, 'District']),
                         (x[i], y[i]),
                         textcoords="offset points", xytext=(4,4), fontsize=8, alpha=0.75)

    plt.grid(alpha=0.3)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()

# Running Graph
plot_scatter_with_fit(df['FiscalIndependence'], df['CrimeRate'],
                      x_label='Fiscal Independence (%)',
                      savepath=None)

plot_scatter_with_fit(df['ReligionRatio'], df['CrimeRate'],
                      x_label='Religious Population Ratio (%)',
                      savepath=None)

plot_scatter_with_fit(df['VolunteerRate'], df['CrimeRate'],
                      x_label='Volunteer Participation Rate (%)',
                      savepath=None)

# Multiple Regression
X = df[['FiscalIndependence', 'ReligionRatio', 'VolunteerRate']].copy()
y = df['CrimeRate'].astype(float)

# Adding Constant term
X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()

print('\n[OLS Summary]')
print(ols_model.summary())

# Coefficient Bar Graph
betas = ols_model.params.drop(labels='const')
errs = ols_model.bse.drop(labels='const')

plt.figure()
indices = np.arange(len(betas))
plt.bar(indices, betas.values, yerr=1.96*errs.values, align='center', alpha=0.9)
plt.xticks(indices, betas.index, rotation=20)
plt.ylabel('Coefficient (β)')
plt.title('OLS Coefficients with 95% CI')
plt.grid(axis='y', alpha=0.3)
plt.show()

# Correlation Matrix
corr_cols = ['CrimeRate','FiscalIndependence','ReligionRatio','VolunteerRate']
C = df[corr_cols].corr().values

fig, ax = plt.subplots()
im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr_cols)))
ax.set_yticks(np.arange(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=30, ha='right')
ax.set_yticklabels(corr_cols)
plt.title('Correlation Matrix')
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f'{C[i,j]:.2f}', ha='center', va='center', color='black', fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
