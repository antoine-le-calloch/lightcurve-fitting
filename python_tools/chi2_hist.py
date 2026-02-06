import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('parametric_timescale_parameters.csv')

if 'chi2' in df.columns:
    all_chi2 = df['chi2'].dropna()
else:
    print("No data found")

total_count = len(all_chi2)

upper_limit = 10
mask_in_range = (all_chi2 >= 0) & (all_chi2 <= upper_limit)

data_to_plot = all_chi2[mask_in_range]

count_outside = total_count - len(data_to_plot)
percent_outside = (count_outside / total_count) * 100 if total_count > 0 else 0

print(f"Total values: {total_count}")
print(f"Values shown (0-{upper_limit}): {len(data_to_plot)}")
print(f"Percentage of values outside this range: {percent_outside:.2f}%")

plt.figure(figsize=(10, 6))
plt.hist(data_to_plot, bins='auto', color='skyblue', edgecolor='black', alpha=0.7)

plt.title(r'$\chi^2$ values')
plt.xlabel(r'$\chi^2$') 
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()