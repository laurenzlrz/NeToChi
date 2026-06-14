from typing import TypeVar

from netochi.input_generator.interfaces import MappingInput
from netochi.mapping.interfaces import MappingState


Input_contra = TypeVar("Input_contra", bound=MappingInput, contravariant=True)
MappingState_contra = TypeVar("MappingState_contra", bound=MappingState, contravariant=True)
BaselineState_contra = TypeVar("BaselineState_contra", bound=MappingState, contravariant=True)
Input_co = TypeVar("Input_co", bound=MappingInput, covariant=True)
MappingState_co = TypeVar("MappingState_co", bound=MappingState, covariant=True)
BaselineState_co = TypeVar("BaselineState_co", bound=MappingState, covariant=True)
