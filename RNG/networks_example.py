
import processRNG

# This example requires an existing set of distributions. Please see distributions_example.py.


if __name__ == "__main__":

    processRNG.generate_networks(

        directory='models',
        group_name='test_group',
        overwrite=False,
    )
