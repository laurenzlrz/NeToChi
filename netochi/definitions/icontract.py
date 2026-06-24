import functools
from enum import IntEnum
from typing import Callable, Any, Type, TypeVar

import icontract

T = TypeVar("T", bound=Callable[..., Any])

class Criticality(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# 2. Global Configuration
class ContractConfig:
    # Only enforce contracts at or above this level
    GLOBAL_THRESHOLD: Criticality = Criticality.MEDIUM

    @classmethod
    def set_threshold(cls, level: Criticality) -> None:
        cls.GLOBAL_THRESHOLD = level


# 3. Helper to intercept contract violations
def _wrap_contract_check(
    contract_decorator_factory: Callable[..., Any],
    condition: Callable[..., bool],
    criticality: Criticality,
    *args: Any,
    **kwargs: Any
) -> Callable[[T], T]:
    """
    Wraps standard icontract decorators. If the global threshold is higher
    than the contract's criticality, the check is bypassed or softened.
    """
    contract_decorator = contract_decorator_factory(condition, *args, **kwargs)

    # This is called by python at preloading
    def decorator(func: T) -> T:

        contract_decorated_func = contract_decorator(func)

        def rt_wrapper(*args: Any, **kwargs: Any) -> Any:
            if criticality < ContractConfig.GLOBAL_THRESHOLD:
                return func(*args, **kwargs)
            return contract_decorated_func(*args, **kwargs)

        return rt_wrapper

    return decorator


def func_require(condition: Callable[..., bool], criticality: Criticality = Criticality.LOW, *args: Any, **kwargs: Any) -> Callable[[T], T]:
    """Extended icontract.require with criticality levels."""
    return _wrap_contract_check(icontract.require, condition, criticality, *args, **kwargs)


def func_ensure(condition: Callable[..., bool], criticality: Criticality = Criticality.LOW, *args: Any, **kwargs: Any) -> Callable[[T], T]:
    """Extended icontract.ensure with criticality levels."""
    return _wrap_contract_check(icontract.ensure, condition, criticality, *args, **kwargs)

def class_invariant(condition: Callable[..., bool], criticality: Criticality = Criticality.LOW, *args: Any, **kwargs: Any) -> Callable[[T], T]:
    """Extended icontract.invariant with criticality levels."""

    @functools.wraps(condition)
    def dynamic_condition(*cond_args: Any, **cond_kwargs: Any) -> bool:
        if criticality < ContractConfig.GLOBAL_THRESHOLD:
            return True
        return condition(*cond_args, **cond_kwargs)

    return icontract.invariant(dynamic_condition, *args, **kwargs)