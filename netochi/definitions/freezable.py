from netochi.definitions.exceptions import FrozenError


class Freezable:
    """Base class for objects that support freeze/unfreeze state management."""

    def freeze(self) -> None:
        pass

    def unfreeze(self) -> None:
        pass


def freezable(cls):
    orig_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        self.__dict__['_is_frozen'] = False
        orig_init(self, *args, **kwargs)
        
    def freeze(self):
        self.__dict__['_is_frozen'] = True
        
    def unfreeze(self):
        self.__dict__['_is_frozen'] = False
        
    def new_setattr(self, name, value):
        if getattr(self, '_is_frozen', False):
            raise FrozenError("Object is frozen!")
        object.__setattr__(self, name, value)

    cls.__init__ = new_init
    cls.freeze = freeze
    cls.unfreeze = unfreeze
    cls.__setattr__ = new_setattr
    return cls
