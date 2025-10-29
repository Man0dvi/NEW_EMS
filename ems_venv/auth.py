# This could go in your existing `auth.py`
def user_has_persona(user, required_persona: str) -> bool:
    return required_persona in getattr(user, "personas", [])
