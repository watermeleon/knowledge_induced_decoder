from torch.utils.data import IterableDataset  

class Example(object):
    """Defines a single training or test example.
    Stores each column of the example as an attribute.
    """
    @classmethod
    def fromdict(cls, data):
        ex = cls(data)
        return ex

    def __init__(self, data):
        for key, val in data.items():
            super(Example, self).__setattr__(key, val)

    def __setattr__(self, key, value):
        raise AttributeError

    def __hash__(self):
        return hash(tuple(x for x in self.__dict__.values()))

    def __eq__(self, other):
        this = tuple(x for x in self.__dict__.values())
        other = tuple(x for x in other.__dict__.values())
        return this == other

    def __ne__(self, other):
        return not self.__eq__(other)



class Dataset(object):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        return data

    # def __len__(self):
    #     return len(self.examples)

    def __getattr__(self, attr):
        for x in self.examples:
            return getattr(x, attr)



def show_eg():
    # examples = [Example.fromdict({'image':"veryimportantpath", 'text': "cute caption of a dog", 'img_id' : 55})]
    examples = [3,4,5]
    egdataset= Dataset(examples)
    print("Is it an IterableDataset?", isinstance(egdataset, IterableDataset))


if __name__ == '__main__':
    show_eg()

