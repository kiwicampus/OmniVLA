def get_vla_dataset_and_collator(*args, **kwargs):
    # Import lazily so callers that only need tokenizers/constants do not eagerly pull in RLDS dataset modules.
    from .materialize import get_vla_dataset_and_collator as _get_vla_dataset_and_collator

    return _get_vla_dataset_and_collator(*args, **kwargs)


__all__ = ["get_vla_dataset_and_collator"]
