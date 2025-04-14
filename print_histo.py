import matplotlib.pyplot as plt
import numpy as np
import random
#palette: https://coolors.co/palette/e15840-e71e24-f38b8c-79af5d-b3cf34-e49343-4196cb-81bde3-8776b6-a494c5

random_value = random.randint(0,1000000)
# Data
groups = [r'$γ = 0\%$', r'$γ = 25\%$', r'$γ = 50\%$', r'$γ = 75\%^\dagger$']
x = np.arange(len(groups))  # X locations for groups
width = 0.165  # Narrower width to fit 4 bars

# Accuracy values
individual = [72.63, 71.50, 71.78, 70.59]
ours = [66.13, 66.00, 66.10, 65.36]
ours_150 = [69.17, 69.25, 69.10, 61.65]
neumeta = [49.03, 48.69, 48.65, 45.38]

# Colors in hex
colors = {
    'individual': '#E15840',    
    'ours': '#E49343',      
    'ours_150': '#79AF5D',          
    'neumeta': '#4196CB'        
}

# Create plot
fig, ax = plt.subplots(figsize=(5,3))
# Spacing multiplier between bars
spacing = 1.2
offsets = [-1.5, -0.5, 0.5, 1.5]
offsets = [o * width * spacing for o in offsets]

# Plot bars
bar1 = ax.bar(x + offsets[0], individual, width, label='Individual', color=colors['individual'], edgecolor='black')
bar2 =ax.bar(x + offsets[1], ours, width, label='Ours', color=colors['ours'], edgecolor='black')
bar3 =ax.bar(x + offsets[2], ours_150, width, label='Ours + 150 epochs per block', color=colors['ours_150'], edgecolor='black')
bar4 =ax.bar(x + offsets[3], neumeta, width, label='NeuMeta', color=colors['neumeta'], edgecolor='black')


# Labels and formatting
ax.set_ylabel('Accuracy (%)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=16)
ax.tick_params(axis='y', labelsize=16)  # Set y-axis tick font size

handles = [bar1[0], bar3[0], bar2[0], bar4[0]]
labels = ['Individual', 'Ours', 'Ours + 150 epochs per block', 'NeuMeta']
fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, frameon=True, fontsize=12)

# Style adjustments
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle=':', linewidth=1)
ax.set_ylim(25, 75)  # Optional, adjust for better visibility

plt.tight_layout()
plt.subplots_adjust(bottom=0.35)  # Make space for the legend
plt.savefig(f'/homes/tsommariva/neumeta/assets/histogram_{random_value}.pdf', dpi=1200, bbox_inches='tight')
print(f'histogram_{random_value}')