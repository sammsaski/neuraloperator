import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os


fno_darcy_losses = [0.5957, 0.3122, 0.2466, 0.2126, 0.1911, 0.1736, 0.1816, 0.1687, 0.1530, 0.1534, 0.1718, 0.1548, 0.1587, 0.1389, 0.1127, 0.1160, 0.1269, 0.1175, 0.1243, 0.1013]
sino_darcy_losses = [0.9059, 0.5903, 0.5247, 0.4604, 0.4204, 0.4233, 0.3784, 0.3645, 0.3537, 0.3378, 0.3301, 0.3268, 0.3197, 0.3072, 0.2967, 0.2920, 0.2880, 0.2825, 0.2744, 0.2700]
fno_burger_losses = [0.4339, 0.0899, 0.0585, 0.0692, 0.0476, 0.0484, 0.0437, 0.0377, 0.0348, 0.0277, 0.0249, 0.0288, 0.0279, 0.0292, 0.0212, 0.0192, 0.0170, 0.0145, 0.0126, 0.0109]
sino_burger_losses = [1.8979, 0.8424, 0.4712, 0.4171, 0.3793, 0.3357, 0.3135, 0.3112, 0.2938, 0.2949, 0.2838, 0.2791, 0.2552, 0.2579, 0.2467, 0.2325, 0.2245, 0.2182, 0.2105, 0.1990]

# fno losses
fno_darcy_16_h1_losses = [0.3940, 0.2872, 0.2250, 0.2144, 0.2013, 0.2239, 0.1983, 0.2138, 0.1985, 0.2475, 0.2110, 0.2406, 0.2184, 0.2065, 0.2083, 0.2333, 0.1990, 0.2028, 0.2109, 0.1938]
fno_darcy_16_l2_losses = [0.2720, 0.2069, 0.1590, 0.1496, 0.1388, 0.1585, 0.1339, 0.1435, 0.1290, 0.1639, 0.1411, 0.1686, 0.1357, 0.1336, 0.1351, 0.1367, 0.1230, 0.1215, 0.1242, 0.1142]
fno_darcy_32_h1_losses = [0.5327, 0.4594, 0.4407, 0.4525, 0.4435, 0.4478, 0.4595, 0.4890, 0.4690, 0.4744, 0.4571, 0.4984, 0.5052, 0.4807, 0.4815, 0.4802, 0.4939, 0.4831, 0.5333, 0.5126]
fno_darcy_32_l2_losses = [0.2768, 0.2261, 0.1879, 0.1833, 0.1714, 0.1874, 0.1769, 0.1875, 0.1665, 0.1918, 0.1706, 0.2021, 0.1749, 0.1690, 0.1713, 0.1727, 0.1665, 0.1606, 0.1732, 0.1621]

fno_burgers_16_h1_losses = [0.1498, 0.0419, 0.0739, 0.0463, 0.0429, 0.0596, 0.0332, 0.0360, 0.0237, 0.0168, 0.0245, 0.0359, 0.0368, 0.0205, 0.0135, 0.0165, 0.0126, 0.0174, 0.0156, 0.0153]
fno_burgers_16_l2_losses = [0.1209, 0.0326, 0.0655, 0.0444, 0.0503, 0.0582, 0.0331, 0.0348, 0.0236, 0.0197, 0.0210, 0.0355, 0.0388, 0.0165, 0.0148, 0.0162, 0.0133, 0.0170, 0.0180, 0.0145]

# sino losses
sino_darcy_16_h1_losses = [0.7393, 0.6755, 0.5319, 0.4954, 0.4862, 0.4191, 0.3890, 0.4020, 0.3476, 0.3573, 0.3529, 0.3544, 0.3369, 0.3140, 0.3125, 0.2971, 0.3072, 0.3035, 0.2891, 0.2842]
sino_darcy_16_l2_losses = [0.4142, 0.3480, 0.2486, 0.2507, 0.2614, 0.2108, 0.2031, 0.2212, 0.1884, 0.2024, 0.2010, 0.1976, 0.1929, 0.1736, 0.1757, 0.1652, 0.1710, 0.1708, 0.1600, 0.1593]
sino_darcy_32_h1_losses = [1.4828, 1.5012, 2.6396, 3.4801, 3.3281, 2.7055, 2.1464, 2.1746, 3.1515, 2.4831, 2.1132, 2.3944, 1.9651, 1.9258, 1.9935, 1.9853, 2.1933, 2.2717, 2.4480, 2.5387]
sino_darcy_32_l2_losses = [0.5848, 0.7327, 1.1198, 1.4501, 1.3957, 1.1113, 0.8843, 0.8962, 1.3365, 1.0755, 0.8980, 1.0209, 0.8448, 0.7873, 0.7838, 0.8043, 0.9116, 0.9329, 0.9919, 1.0118]

sino_burgers_16_h1_losses = [0.7396, 0.5137, 0.4486, 0.4080, 0.3504, 0.3293, 0.3033, 0.3020, 0.3263, 0.2829, 0.3208, 0.2647, 0.2533, 0.2555, 0.2622, 0.2302, 0.2266, 0.2214, 0.2042, 0.1968]
sino_burgers_16_l2_losses = [0.5039, 0.2755, 0.2329, 0.2134, 0.1776, 0.1864, 0.1389, 0.1526, 0.1912, 0.1408, 0.1688, 0.1397, 0.1158, 0.1253, 0.1488, 0.1049, 0.1084, 0.1022, 0.0923, 0.0904]


def plot_method_comparison(
    data_dict,
    title="Loss Comparison Across Methods and Resolutions",
    ylabel="Loss",
    xlabel="Epoch",
    figsize=(10, 6),
    save_path=None
):
    """
    Plot multiple loss curves for two methods across resolutions and metrics.

    Parameters
    ----------
    data_dict : dict
        Structure:
        {
            "Method A": {
                "16_h1": [...],
                "16_l2": [...],
                "32_h1": [...],
                "32_l2": [...]
            },
            "Method B": {
                ...
            }
        }
    title : str
        Title of the plot.
    ylabel : str
        Label for Y axis.
    xlabel : str
        Label for X axis.
    figsize : tuple
        Size of the plot.
    save_path : str, optional
        If given, saves the plot instead of showing it.
    """
    plt.figure(figsize=figsize)

    # Color/style map to keep things visually distinct
    styles = {
        "16_h1": ("-", "16 H1"),
        "16_l2": ("--", "16 L2"),
        "32_h1": ("-.", "32 H1"),
        "32_l2": (":", "32 L2")
    }

    for method_name, losses in data_dict.items():
        for key, values in losses.items():
            if key not in styles:
                continue
            style, label_suffix = styles[key]
            label = f"{method_name} ({label_suffix})"
            plt.plot(values, linestyle=style, linewidth=2, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()


def plot_single_resolution_comparison(
    data_dict,
    title="Loss Comparison (Resolution = 16)",
    ylabel="Loss",
    xlabel="Epoch",
    figsize=(10, 6),
    save_path=None
):
    """
    Plot loss curves for two methods at a single resolution (16),
    comparing both H1 and L2 losses.

    Parameters
    ----------
    data_dict : dict
        Structure:
        {
            "Method A": {
                "h1": [...],
                "l2": [...]
            },
            "Method B": {
                ...
            }
        }
    title : str
        Title of the plot.
    ylabel : str
        Label for Y axis.
    xlabel : str
        Label for X axis.
    figsize : tuple
        Size of the plot.
    save_path : str, optional
        If given, saves the plot instead of showing it.
    """
    plt.figure(figsize=figsize)

    styles = {
        "h1": ("-", "H1"),
        "l2": ("--", "L2")
    }

    for method_name, losses in data_dict.items():
        for key, values in losses.items():
            if key not in styles:
                continue
            style, label_suffix = styles[key]
            label = f"{method_name} ({label_suffix})"
            plt.plot(values, linestyle=style, linewidth=2, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()


def make_gif_from_epochs(
    folder_path=".",
    file_prefix="fno_outputs_darcy_16_epoch=",
    file_suffix=".png",
    epoch_range=range(20),
    gif_name="fno_outputs_darcy_16.gif",
    duration=300,
    loop=0,
    font_size=20
):
    """
    Create a GIF from a sequence of PNGs labeled by epoch, with epoch number drawn on each frame.

    Parameters
    ----------
    folder_path : str
        Directory where the images are stored.
    file_prefix : str
        Prefix of the image files before the epoch number.
    file_suffix : str
        Suffix of the image files (e.g., '.png').
    epoch_range : iterable
        Epoch numbers to include in order (e.g., range(20)).
    gif_name : str
        Name of the output GIF file.
    duration : int
        Delay between frames in milliseconds.
    loop : int
        Number of times the GIF should loop (0 = infinite).
    font_size : int
        Font size of the epoch label.
    """
    image_paths = [
        os.path.join(folder_path, f"{file_prefix}{i}{file_suffix}")
        for i in epoch_range
    ]

    # Check all files exist
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing image: {path}")

    frames = []

    # Try loading default PIL font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")  # Ensure RGB mode
        draw = ImageDraw.Draw(img)
        text = f"epoch = {epoch_range[i]}"
        draw.text((10, 10), text, font=font, fill=(255, 0, 0))  # Red text
        frames.append(img)

    gif_path = os.path.join(folder_path, gif_name)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )

    print(f"✅ GIF saved to {gif_path}")



if __name__ == "__main__":
    darcy_loss_dict = {
        "FNO": {
            "16_h1": fno_darcy_16_h1_losses,
            "16_l2": fno_darcy_16_l2_losses,
            "32_h1": fno_darcy_32_h1_losses,
            "32_l2": fno_darcy_32_l2_losses
        },
        "SINO": {
            "16_h1": sino_darcy_16_h1_losses,
            "16_l2": sino_darcy_16_l2_losses,
            "32_h1": sino_darcy_32_h1_losses,
            "32_l2": sino_darcy_32_l2_losses
        }
    }

    burgers_loss_dict = {
        "FNO": {
            "h1": fno_burgers_16_h1_losses,
            "l2": fno_burgers_16_l2_losses,
        },
        "SINO": {
            "h1": sino_burgers_16_h1_losses,
            "l2": sino_burgers_16_l2_losses,
        }
    }

    
    plot_method_comparison(darcy_loss_dict, save_path="Darcy_loss_analysis.png")
    plot_single_resolution_comparison(burgers_loss_dict, save_path="Burgers_loss_analysis.png")
    
    # make fno gifs
    make_gif_from_epochs("../plots/fno/16", "fno_outputs_darcy_16_epoch=", gif_name="fno_outputs_darcy_16.gif")
    make_gif_from_epochs("../plots/fno/32", "fno_outputs_darcy_32_epoch=", gif_name="fno_outputs_darcy_32.gif")
    make_gif_from_epochs("../plots/sino/16", "sino_outputs_darcy_16_epoch=", gif_name="sino_outputs_darcy_16.gif")
    make_gif_from_epochs("../plots/sino/32", "sino_outputs_darcy_32_epoch=", gif_name="sino_outputs_darcy_32.gif")