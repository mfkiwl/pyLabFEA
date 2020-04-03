# pyLabFEA

### Python Laboratory for Finite Element Analysis

  - Author: Alexander Hartmaier
  - Organization: ICAMS, Ruhr University Bochum, Germany
  - Contact: <alexander.hartmaier@rub.de>

Finite Element Analysis (FEA) is a numerical method for studying
mechanical behavior of fluids and solids. The pyLabFEA package
introduces a simple version of FEA for solid mechanics and
elastic-plastic materials, which is fully written in Python. Due to
its simplicity, it is well-suited for teaching, and its flexibility in
constitutive modeling of materials makes it a useful research tool.

## Microstructure Branch

This branch of the pyLabFEA package is intended for new developments to explicitly 
take the microstructure of a material into account when defining its mechanical
properties. This branch is still under construction. Please refer to the documentation 
in the master branch.

##Installation

The pyLabFEA package is installed with the following command
```python
$ python setup.py install --user

```
After this, the package can by imported with

```
import pylabfea as fea

```
## Documentation

Online documentation for pyLabFEA can be found under https://ahartmaier.github.io/pyLabFEA/.
For offline use, open pyLabFEA/docs/index.html to browse through the contents.
The documentation is generated using [Sphinx](http://www.sphinx-doc.org/en/master/).

## Contributions

Contributions to the pyLabFEA package are highly welcome, either in form of new 
notebooks with application examples or tutorials, or in form of new functionalities 
to the Python code. Furthermore, bug reports or any comments on possible improvements of 
the code or its documentation are greatly appreciated.

## Dependencies

pyLabFEA requires the following packages as imports:

 - [NumPy](http://numpy.scipy.org) for array handling
 - [Scipy](https://www.scipy.org/) for numerical solutions
 - [scikit-learn](https://scikit-learn.org/stable/) for machine learning algorithms
 - [MatPlotLib](https://matplotlib.org/) for graphical output
 - [pandas](https://pandas.pydata.org/) for data import

## License

The pyLabFEA package comes with ABSOLUTELY NO WARRANTY. This is free
software, and you are welcome to redistribute it under the conditions of
the GNU General Public License
([GPLv3](http://www.fsf.org/licensing/licenses/gpl.html))

The contents of the examples and notebooks are published under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
([CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/))
