# SCAFF

Power and EM side channels in Python.

Side-Channel Analysis Flexible Framework (SCAFF) initially aimed at being a Python library and command-line tool to perform power or electromagnetic side-channel attacks.
Started from a dirty research project, its usage is currently considered as working but unstable.

**References**

- **[phase_data](https://github.com/pierreay/phase_data.git)**: PhaseSCA research project leveraging this tool.
- **[screaming_channels_ble](https://github.com/pierreay/screaming_channels_ble)**: BlueScream research project from which this tool is born.

# Installation

Installable using [Pip](https://pypi.org/project/pip/):

```bash
git clone https://github.com/pierreay/scaff.git
cd scaff && pip install --user .
```

Optionally, install [Histogram Enumeration Library (HEL)](https://github.com/pierreay/python_hel) for key estimation and enumeration.
