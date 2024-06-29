"""Plotting functions and classes."""

# * Importation

# Standard import.

# External import.
import matplotlib.pyplot as plt

# Internal import.

# * Global configuration

# Use a standard bright Matplotlib style.
plt.style.use('bmh')

# Number of bins for the FFT.
NFFT = 256

# Flag indicated that LaTeX fonts have been enabled.
LATEX_FONT_ENABLED = False

# * Functions

def enable_latex_fonts():
    """Use LaTeX for text rendering."""
    global LATEX_FONT_ENABLED
    # Use pdflatex to generate fonts.
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15
    })
    LATEX_FONT_ENABLED = True

# * Classes
