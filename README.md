# SCAFF

Power and EM side channels in Python.

Side-Channel Analysis Flexible Framework (SCAFF) initially aimed at being a Python library and command-line tool to perform power or electromagnetic side-channel attacks.
Started from a dirty research project, its usage is currently considered as working but unstable.

**References**

- **[phasesca](https://github.com/pierreay/phasesca.git)**: PhaseSCA research project leveraging this tool.
- **[bluescream](https://github.com/pierreay/bluescream)**: BlueScream research project from which this tool is born.

# Installation

Installable using [Pip](https://pypi.org/project/pip/):

```bash
git clone https://github.com/pierreay/scaff.git
cd scaff && pip install --user .
```

Optionally, install [Histogram Enumeration Library (HEL)](https://github.com/giocamurati/python_hel) for key estimation and enumeration.
