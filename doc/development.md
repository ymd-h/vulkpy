# Developer's Guide and Contributing


## docstring
To make the usage understandable, all public classes and methods
should have docstring.

[Shpinx](https://www.sphinx-doc.org/) generates
[API reference](https://ymd-h.github.io/vulkpy/api.html)
from these docstring.


Basically we obey
[Numpy's style guide](https://numpydoc.readthedocs.io/en/latest/format.html),
however, we adopt following [PEP-257](https://peps.python.org/pep-0257/)
statement for class docstring;

> The docstring for a class should summarize its behavior and list the
> public methods and instance variables. If the class is intended to
> be subclassed, and has an additional interface for subclasses, this
> interface should be listed separately (in the docstring). The class
> constructor should be documented in the docstring for its __init__
> method. Individual methods should be documented by their own
> docstring.


To separate class docstring and `__init__()` docstring,
we configure Sphinx as follows;

```python:conf.py
autodoc_class_signature = "separated"
autodoc_default_options = {
    "class-doc-from": "class"
}
```
