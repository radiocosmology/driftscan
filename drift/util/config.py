"""
A module to define strictly typed attributes of a class, than can be loaded
from an input dictionary. This is particularly useful for loading a class from
a YAML document.


In this example we set up a class to store information about a person.

    >>> class Person(Reader):
    ...
    ...     name = Property(default='Bill', proptype=str, key='name')
    ...     age = Property(default=26, proptype=float, key='ageinyears')

We then extend it to store information about a person with a pet. The
configuration will be successfully inherited.

    >>> class PersonWithPet(Person):
    ... 
    ...     petname = Property(default='Molly', proptype=str, key='dog')

Let's create a couple of objects from these classes.

    >>> person1 = Person()
    >>> person2 = Person()

And a dictionary of replacement parameters.

    >>> testdict = { 'myname' : 'Richard', 'ageinyears' : 40, 'petname' : 'Sooty'}

First let's check what the default parameters are:

    >>> print person1.name, person1.age
    Bill 26.0
    >>> print person2.name, person2.age, person2.dog
    Bill 26.0 Molly

Now let's load the configuration from a dictionary:

    >>> person1.fromconfig(testdict)
    Richard 40.0

    >>> person2.fromconfig(testdict)
    Richard 40.0 Sooty
    
Then we'll print the output to see the updated configuration:

    >>> print person1.name, person1.age
    Richard 40.0
    >>> print person2.name, person2.age, person2.dog
    Richard 40.0 Sooty
"""


class Property(object):
    """Custom property descriptor that can load values from a given dict.
    """

    def __init__(self, default=None, proptype=None, key=None):
        """Make a new property type.

        Parameters
        ----------
        default : object
            The initial value for the property.
        proptype : function
            The type of the property. In reality this is just a function which
            gets called whenever we update the value: `val = proptype(newval)`
        key : string
            The name of the dictionary key that we can fetch this value from.
            If None (default), attempt to use the attribute name from the
            class.
        """
        import weakref

        self.proptype = (lambda x: x) if proptype is None else proptype
        self.default = default
        self.key = key

        ## As the instantiated Property exists per class we store the current
        ## values in a dictionary of instances.
        self.instances = weakref.WeakKeyDictionary()

    def __get__(self, obj, objtype):
        ## Object getter.
        if obj is None:
            return None

        if obj not in self.instances:
            self.instances[obj] = self.proptype(self.default)

        return self.instances[obj]

    def __set__(self, obj, val):
        ## Object setter.
        if obj is None:
            return None

        self.instances[obj] = self.proptype(val)

    def _from_config(self, obj, config, key=None):
        """Load the configuration from the supplied dictionary.

        Parameters
        ----------
        obj : object
            The parent object of the Property that we want to update.
        config : dict
            Dictionary of configuration values.
        """
        if self.key is None and key is not None:
            self.key = key

        if self.key in config:
            val = config[self.key]
            self.instances[obj] = self.proptype(val)
            print "Setting attribute %s to %s." % (key, val)


class Reader(object):
    """A class that allows the values of Properties to be assigned from a dictionary.
    """

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        """Create a new instance with values loaded from config.

        Parameters
        ----------
        config : dict
            Dictionary of configuration values.
        """

        c = cls(*args, **kwargs)
        c.read_config(config)

        return c


    def read_config(self, config):
        """Set all properties in this class from the supplied config.

        Parameters
        ----------
        config : dict
            Dictionary of configuration values.
        """
        import inspect

        for basecls in inspect.getmro(type(self))[::-1]:
            for propname, clsprop in basecls.__dict__.items():
                if isinstance(clsprop, Property):
                    clsprop._from_config(self, config, key=propname)

