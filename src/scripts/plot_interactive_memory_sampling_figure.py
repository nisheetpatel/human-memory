import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.special import erf
from scipy.stats import norm


def create_interactive_plot():
    # Set up the figure and subplots
    plt.rcParams.update({'font.size': 18})  # Increase overall font size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.85, top=0.95, hspace=0.4)

    # Parameters
    mu = 0
    sigma = 1
    rho = 1  # Default price as requested

    # Generate x values
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

    # Gaussian distribution
    y = norm.pdf(x, mu, sigma)

    # Plot Gaussian distribution
    line1, = ax1.plot(x, y, 'g-', lw=3, label='Value distribution')
    ax1.set_title('Value encoded in memory N(μ, σ²)', fontsize=24, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=22)
    ax1.set_ylabel('Probability density', fontsize=22)

    # Remove y-tick labels from top panel
    ax1.set_yticklabels([])

    # Vertical line for mean (mu)
    ax1.axvline(mu, color='g', linestyle=':', lw=2, label='Mean (μ)')

    # Vertical line for price
    price_line1 = ax1.axvline(rho, color='r', linestyle='--', lw=2, label='Price (ρ)')

    # Shaded area
    fill1 = ax1.fill_between(x, 0, y, where=(x > rho), alpha=0.3, color='orange', label='p(μ > ρ)')

    # Add delta annotation
    delta = rho - mu
    ax1.annotate('', xy=(mu, 0.05), xytext=(rho, 0.05),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax1.text((mu + rho) / 2, 0.07, '$\\delta$', color='blue', ha='center', va='bottom', fontsize=20)

    # Psychometric curve
    x_psychometric = np.linspace(-4*sigma, 4*sigma, 1000)
    y_psychometric = 0.5 * (1 + erf((x_psychometric - (mu - rho)) / (sigma * np.sqrt(2))))

    # Plot psychometric curve
    line2, = ax2.plot(x_psychometric, y_psychometric, 'b-', lw=3)
    ax2.set_title('Psychometric Curve', fontsize=24, fontweight='bold')
    ax2.set_xlabel('Expected reward (μ - ρ)', fontsize=22)
    ax2.set_ylabel('$p(Response = Yes)$', fontsize=22)
    ax2.set_ylim(0, 1)

    # Remove x-tick labels from bottom panel
    ax2.set_xticklabels([])

    # Vertical line for current expected reward
    price_line2 = ax2.axvline(mu - rho, color='r', linestyle='--', lw=2)

    # Add legends with larger font
    ax1.legend(fontsize=18)
    ax2.legend(['Psychometric curve', 'Expected reward'], fontsize=18)

    # Add slider for price
    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax, 'Price (ρ)', mu - 3*sigma, mu + 3*sigma, valinit=rho, valstep=0.01)
    slider.label.set_fontsize(20)

    def update(val):
        rho = slider.val
        delta = rho - mu
        
        # Update price line in Gaussian plot
        price_line1.set_xdata([rho, rho])
        
        # Update shaded area
        ax1.collections.clear()
        ax1.fill_between(x, 0, y, where=(x > rho), alpha=0.3, color='orange')
        
        # Update delta annotation
        ax1.texts[-1].set_position(((mu + rho) / 2, 0.07))
        ax1.texts[-1].set_text('$\\delta$')
        ax1.annotations[-1].xy = (rho, 0.05)
        
        # Update psychometric curve
        y_psychometric_new = 0.5 * (1 + erf((x_psychometric - (mu - rho)) / (sigma * np.sqrt(2))))
        line2.set_ydata(y_psychometric_new)
        
        # Update expected reward line
        price_line2.set_xdata([mu - rho, mu - rho])
        
        # Update x-ticks for top panel
        ax1.set_xticks([mu, rho])
        ax1.set_xticklabels(['μ', 'ρ'])
        
        # Update x-ticks for bottom panel
        ax2.set_xticks([-delta, 0, delta])
        ax2.set_xticklabels(['$-\\delta$', '0', '$\\delta$'])
        
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Initial setup of x-ticks
    ax1.set_xticks([mu, rho])
    ax1.set_xticklabels(['μ', 'ρ'])
    ax2.set_xticks([-delta, 0, delta])
    ax2.set_xticklabels(['$-\\delta$', '0', '$\\delta$'])

    # Improve overall aesthetics
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=18)

    return fig, slider

# Run the function to create the interactive plot
fig, slider = create_interactive_plot()

# Display the plot
plt.savefig("DRA_memory_sampling_figure.svg")
plt.savefig("DRA_memory_sampling_figure.png")
plt.close()