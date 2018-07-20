def visualize (iq_lines, title='', equalize=False, min_dB=None, save_path=None, figsize=(12, 6), aspect_ratio='auto'):
    import matplotlib.pyplot as plt
    import numpy as np

    num_samples, num_lines = iq_lines.shape
    center_magnitude = abs (iq_lines[:, num_lines//2].real)
    # Detect envelope
    data = abs (iq_lines)
    # Full-scale contrast stretch into range [0, 1]
    low, high = data.min (), data.max ()
    data = (data - low) / (high - low)
    # Histogram equalize
    if equalize:
        hist, bins = np.histogram (abs (data), bins=256)
        cdf = np.cumsum (hist)
        cdf = cdf/cdf[-1]
        data = np.interp (data, bins[:-1], cdf)
    # Log compress into range [min_dB, 0] dB
    if min_dB is not None:
        data = 10 * np.log10(data + 10**(min_dB/10))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    plt.sca (ax)
    plt.plot(center_magnitude, color=(153/255,102/255,204/255))
    plt.xlabel ('Depth', fontsize=14, labelpad=15)
    plt.ylabel ('Amplitude', fontsize=14, labelpad=15)
    plt.yticks ([])
    plt.grid ()
    plt.title ('Center RF-Line Magnitude', fontsize=16, pad=15)

    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible (False)

    ax = axes[1]
    plt.sca (ax)
    image = plt.imshow (data, cmap='gray', interpolation='nearest')
    ax.set_aspect (aspect_ratio)
    plt.xlabel ('Width', fontsize=14, labelpad=15)
    plt.ylabel ('Depth', fontsize=14, labelpad=15)
    if title:
        plt.title (title, fontsize=16, pad=15)
    plt.grid ()

    plt.tick_params (axis='both', which='both', bottom=True, top=False,
                    labelbottom=True, left=True, right=False, labelleft=True)
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_visible (False)
    for side in ['bottom', 'left']:
        ax.spines[side].set_position(('outward', 1))

    if min_dB is not None:
        cbar = fig.colorbar (image)
        cbar.set_label ('(dB)', fontsize=12)

    if save_path:
        plt.savefig(save_path)
        print('Image written to disk at:\n{}'.format (save_path))

    plt.show()
