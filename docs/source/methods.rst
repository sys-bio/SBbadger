Methods
=======

Generating a singular model (as a string variable)
--------------------------------------------------

.. autofunction:: SBbadger.generate.model

Generating a collection of Models
-----------------

.. autofunction:: SBbadger.generate.models

Generating Distributions
------------------------

.. autofunction:: SBbadger.generate.distributions

Generating Networks
-------------------

.. autofunction:: SBbadger.generate.networks

Applying Rate-Laws
------------------

.. autofunction:: SBbadger.generate.rate_laws

Standard Networks
-----------------

~~~~~~
Linear
~~~~~~

.. autofunction:: SBbadger.generate.linear

~~~~~~
Cyclic
~~~~~~

.. autofunction:: SBbadger.generate.cyclic

~~~~~~~~
Branched
~~~~~~~~

.. autofunction:: SBbadger.generate.branched

Note: all the above methods, aside from ``generate.model``, use python multiprocessing to generate models or their
components in parallel. To generate them serially use ``generate_serial.<method>``. The only difference is the absence
of the ``n_cpus`` argument.