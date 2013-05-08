import unittest

import pickle

from drift.util import config


class Person(config.Reader):
    name = config.Property(default='Bill', proptype=str)
    age = config.Property(default=26, proptype=float, key='ageinyears')


class PersonWithPet(Person):
    petname = config.Property(default='Molly', proptype=str)
    petage = 36


class TestConfig(unittest.TestCase):

    testdict = { 'name' : 'Richard', 'ageinyears' : 40, 'petname' : 'Sooty'}

    def test_default_params(self):

        person1 = Person()

        self.assertEqual(person1.name, 'Bill')
        self.assertEqual(person1.age, 26.0)
        self.assertIsInstance(person1.age, float)

    def test_set_params(self):

        person = Person()
        person.name = 'Mick'

        self.assertEqual(person.name, 'Mick')

    def test_read_config(self):

        person = Person()
        person.read_config(self.testdict)

        self.assertEqual(person.name, 'Richard')
        self.assertEqual(person.age, 40.0)

    def test_inherit_read_config(self):

        person = PersonWithPet()
        person.read_config(self.testdict)

        self.assertEqual(person.name, 'Richard')
        self.assertEqual(person.age, 40.0)
        self.assertEqual(person.petname, 'Sooty')        

    def test_pickle(self):

        person = PersonWithPet()
        person.read_config(self.testdict)
        person2 = pickle.loads(pickle.dumps(person))

        self.assertEqual(person2.name, 'Richard')
        self.assertEqual(person2.age, 40.0)
        self.assertEqual(person2.petname, 'Sooty')   
