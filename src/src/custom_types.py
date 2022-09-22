import typing


class Array:
    def __class_getitem__(cls, item):
        return typing.Annotated[cls, item]


Batch = typing.TypeVar('Batch', bound=int)
Num = typing.TypeVar('Num', bound=int)
Shape = tuple[typing.TypeVar('Shape', bound=int), ...]  # Bit of a non-PEP hack.
