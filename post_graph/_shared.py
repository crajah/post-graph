"""Logic both clients need, in one place so they cannot disagree about it.

The two clients have diverged before, and always the same way: a rule lived in
one of them and was reimplemented, or not implemented, in the other. ``
add_vertex`` translated a duplicate id and ``upsert_vertex`` did not; the
SQLAlchemy client put its INSERT inside the try and the asyncpg one did not.
Both were invisible because each client's tests only ever exercised that
client.

So anything that is a decision rather than a statement lives here: how an
index name is derived, how an identifier is bounded, how a driver's uniqueness
violation becomes this library's error. The clients keep only the part that is
genuinely about their driver.
"""
import hashlib
import re
from typing import List, Optional, Sequence, Tuple, Union

from post_graph.errors import UniqueConstraintError

#: PostgreSQL's NAMEDATALEN - 1. Identifiers longer than this are truncated
#: silently, which is how two distinct index names sharing a 63-byte prefix
#: become one index -- and the second uniqueness constraint is never created.
MAX_IDENTIFIER_BYTES = 63

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class SharedLogic:
    """Mixin of the rules both clients follow."""

    MAX_IDENTIFIER_BYTES = MAX_IDENTIFIER_BYTES

    # ------------------------------------------------------------ identifiers

    @classmethod
    def _bounded_index_name(cls, stem: str) -> str:
        """*stem*, shortened to fit an identifier if it does not already.

        Truncating alone would let two different key tuples collide, so the
        shortened form carries a hash of the full stem. Deterministic, so
        ``IF NOT EXISTS`` still recognises the index it made last time.
        """
        if len(stem.encode("utf-8")) <= MAX_IDENTIFIER_BYTES:
            return stem
        digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:10]
        keep = MAX_IDENTIFIER_BYTES - len(digest) - 1
        head = stem.encode("utf-8")[:keep].decode("utf-8", "ignore")
        return f"{head}_{digest}"

    @classmethod
    def _assert_identifier_fits(cls, name: str, *, what: str) -> str:
        if len(name.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
            raise ValueError(
                f"{what} {name!r} is {len(name.encode('utf-8'))} bytes; "
                f"PostgreSQL truncates identifiers past {MAX_IDENTIFIER_BYTES} "
                f"and two names sharing a prefix would silently become one "
                f"object, dropping whatever the second one was for. Shorten it."
            )
        return name

    @staticmethod
    def _validate_database_name(name: str, *, what: str) -> str:
        """A bare identifier that fits. Raises ValueError.

        ``CREATE DATABASE`` takes no bound parameter, so the name is
        interpolated and validating it is the only defence there is.
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"{what} must be a non-empty string.")
        if not _IDENTIFIER_RE.match(name):
            raise ValueError(
                f"Invalid {what}: {name!r}. CREATE DATABASE takes no bound "
                f"parameter, so the name is interpolated and must be a bare "
                f"identifier: letters, digits and underscores, starting with a "
                f"letter or underscore."
            )
        return SharedLogic._assert_identifier_fits(name, what=what)

    # ------------------------------------------------------------------ LIKE

    @staticmethod
    def _like_literal(value: str) -> str:
        """Escape a string for use as a literal inside ``LIKE ... ESCAPE '!'``.

        ``_`` matches any character in LIKE, so an unescaped prefix of
        ``learner_`` also matches ``learnerX``. That is not a cosmetic
        difference where the result decides which schemas get dropped.
        """
        return (value.replace("!", "!!")
                     .replace("%", "!%")
                     .replace("_", "!_"))

    # -------------------------------------------------------------------- ids

    @staticmethod
    def _normalise_ids(record_ids) -> List[int]:
        """One id or many, as integers. ``realm/table/42`` yields 42."""
        if record_ids is None:
            return []
        if isinstance(record_ids, (str, int)) or not hasattr(record_ids, "__iter__"):
            record_ids = [record_ids]
        out: List[int] = []
        for raw in record_ids:
            text = str(raw)
            out.append(int(text.split("/")[-1] if "/" in text else text))
        return out

    # ------------------------------------------------------------- uniqueness

    @staticmethod
    def _constraint_name_of(exc: BaseException) -> Optional[str]:
        """The violated constraint, wherever the driver put it.

        asyncpg exposes ``constraint_name`` on the exception. SQLAlchemy wraps
        it, so the original is under ``orig`` or ``__cause__`` -- and the
        wrapper is what application code catches, which is why reading it
        belongs here rather than at each call site.
        """
        for candidate in (exc, getattr(exc, "orig", None),
                          getattr(exc, "__cause__", None)):
            if candidate is None:
                continue
            name = getattr(candidate, "constraint_name", None)
            if name:
                return str(name)
            nested = getattr(candidate, "__cause__", None)
            name = getattr(nested, "constraint_name", None) if nested else None
            if name:
                return str(name)
        # Last resort: PostgreSQL names it in the message text.
        match = re.search(r'unique constraint "([^"]+)"', str(exc))
        return match.group(1) if match else None

    @classmethod
    def _unique_error(cls, exc, *, kind: str, record_id: Optional[str],
                      table_name: str, realm: str) -> UniqueConstraintError:
        """Translate a driver uniqueness violation, saying what was violated.

        The message used to assert the *id* was taken, which was true while the
        primary key was the only uniqueness a table had. With a unique payload
        index -- a username per tenant, a slug per institution -- the id is
        usually fine and something else collided, so a message naming the id
        sends the reader to the wrong place. ``constraint_name`` carries the
        answer either way, so a caller can branch on it without reading
        ``__cause__`` or importing the driver's exception type.
        """
        name = cls._constraint_name_of(exc)
        looks_like_pkey = bool(name) and (
            name.endswith("_pkey") or name.endswith("_pkey1"))
        if name is None or looks_like_pkey:
            where = f"{kind} with ID '{record_id}'" if record_id else kind
            message = (f"{where} already exists in table '{table_name}' "
                       f"under realm '{realm}'.")
        else:
            message = (f"{kind} violates unique constraint '{name}' on table "
                       f"'{table_name}' under realm '{realm}': {exc}")
        return UniqueConstraintError(message, constraint_name=name)

    # ---------------------------------------------------------- index planning

    @classmethod
    def _plan_payload_index(
        cls,
        table_name: str,
        keys: Sequence[str],
        *,
        numeric: bool,
        unique: bool,
        scope_by_space: bool,
        index_name: Optional[str],
        validate_identifier,
    ) -> Tuple[str, List[str]]:
        """The index's name and column list. Shared so the two clients agree.

        A non-unique index over a single key keeps exactly the name and shape
        earlier releases built, so an existing caller does not create a second
        index alongside the one it already has.
        """
        accessor = [f"(((payload->>'{k}'))::numeric)" if numeric
                    else f"((payload->>'{k}'))" for k in keys]

        if len(keys) == 1 and not unique and index_name is None:
            safe = re.sub(r"[^a-z0-9_]", "_", keys[0].lower())
            derived = (f"idx_{table_name}_payload_{safe}"
                       + ("_num" if numeric else ""))
            columns = list(accessor)
        else:
            leading = ["realm", "space"] if scope_by_space else ["realm"]
            columns = leading + accessor
            derived = "_".join(
                [("uq" if unique else "idx"), table_name]
                + (["space"] if scope_by_space else [])
                + [re.sub(r"[^a-z0-9_]", "_", k.lower()) for k in keys]
                + (["num"] if numeric else []))

        if index_name is not None:
            validate_identifier(index_name)
            cls._assert_identifier_fits(index_name, what="index_name")
            return index_name, columns
        return cls._bounded_index_name(derived), columns

    @classmethod
    def _payload_index_keys(cls, key: Union[str, Sequence[str]], *,
                            numeric: bool, validate_payload_key) -> List[str]:
        keys = [key] if isinstance(key, str) else list(key)
        if not keys:
            raise ValueError(
                "create_payload_index needs at least one payload key.")
        for k in keys:
            validate_payload_key(k)
        if numeric and len(keys) > 1:
            raise ValueError(
                "numeric=True applies to every key, which is rarely what a "
                "composite index wants. Build it with one key, or create the "
                "index directly if a mixed composite is really intended."
            )
        return keys
