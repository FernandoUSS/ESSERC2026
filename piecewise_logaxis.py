import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Piecewise LOG transform
# ----------------------------
def custom_log_transform(x):
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan, dtype=float)

    # Region 1
    mask1 = (x >= 1) & (x <= 1e4)
    y[mask1] = np.log10(x[mask1]) / 4.0

    # Region 2
    log_min = np.log10(2e4)
    log_max = np.log10(3e4)

    mask2 = (x >= 2e4) & (x <= 3e4)
    y[mask2] = 1.0 + (np.log10(x[mask2]) - log_min) / (log_max - log_min)

    return y


# ----------------------------
# Data
# ----------------------------
x = np.logspace(0, 5, 3000)
y = np.sin(np.log10(x))
x_t = custom_log_transform(x)

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(5, 3))

mask = ~np.isnan(x_t)
ax.plot(x_t[mask], y[mask], linewidth=1.5)

# ----------------------------
# Major ticks
# ----------------------------
ticks_major_r1 = [1, 10, 1e2, 1e3, 1e4]
ticks_major_r2 = [2e4, 3e4]

ticks_major = ticks_major_r1 + ticks_major_r2
ax.set_xticks(custom_log_transform(ticks_major))
ax.set_xticklabels([
    r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$',
    r'$2\cdot10^4$', r'$3\cdot10^4$'
])

# ----------------------------
# Minor ticks (FIXED)
# ----------------------------

# Region 1: classic log minors
ticks_minor_r1 = []
for decade in [1, 10, 100, 1000]:
    ticks_minor_r1.extend([i * decade for i in range(2, 10)])

# Region 2: generate UNIFORM in log space
log_min = np.log10(2e4)
log_max = np.log10(3e4)

# dense log sampling → then exponentiate
log_ticks = np.linspace(log_min, log_max, 20)
ticks_minor_r2 = 10**log_ticks

ticks_minor = ticks_minor_r1 + list(ticks_minor_r2)
ax.set_xticks(custom_log_transform(ticks_minor), minor=True)

# ----------------------------
# Styling
# ----------------------------
ax.set_xlim(0, 2)

ax.tick_params(axis='x', which='major', length=6, width=1.2)
ax.tick_params(axis='x', which='minor', length=3, width=0.8)

ax.set_xlabel('Piecewise logarithmic axis')
ax.set_ylabel('Signal')

# separator
ax.axvline(1, linestyle='--', color='k', alpha=0.5)

ax.grid(which='major', linestyle='-', alpha=0.3)
ax.grid(which='minor', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()