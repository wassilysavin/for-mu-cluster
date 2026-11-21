"""Conversation persistence, replay and ownership of model responses."""

import hashlib
import hmac
import json
import re
import secrets
import threading
import uuid

from .dialogue import createState, executeTurn, modelHistory
from .functiongemma import decodeCalls
from .model_result import IncompleteModelOutput
from .model_tools import compileModelCalls, modelTools
from .schema import JS_TRIM, is_safe_integer, canonical, clone, ContractError, freeze

_FAILURE = "I couldn't produce a reliable answer to that request."
_CONTEXT_LIMIT = "This conversation is too long to continue. Please start a new chat."


def _initial(registry_id, session_id, version):
    initial = dict(version=1, registry_id=registry_id, session_id=session_id)
    if version == 2:
        initial["dialogue_version"] = 2
    return freeze(dict(initial, semantic=createState(), exchanges=[]))


def createSession(registry, options=None):
    options = options or {}
    version = options.get("dialogueVersion", 1)
    session_id = options["sessionId"] if "sessionId" in options else str(uuid.uuid4())
    if not is_safe_integer(version) or version not in (1, 2) or not isinstance(session_id, str) or not session_id:
        raise ContractError("invalid_session_options", "Invalid dialogue version or session identity.")
    return _initial(registry.fingerprint(), session_id, version)


def recoveryExchanges(session):
    return session["exchanges"] if session.get("dialogue_version") == 2 else None


def _append(session, semantic, exchange):
    record = dict(id=f"e{len(session['exchanges']) + 1}", **exchange)
    return freeze(dict(session, semantic=semantic, exchanges=[*session["exchanges"], record]))


def _current(registry, session):
    if (
        not isinstance(session, dict)
        or isinstance(session.get("version"), bool)
        or session.get("version") != 1
        or ("dialogue_version" in session and session["dialogue_version"] != 2)
        or session.get("registry_id") != registry.fingerprint()
    ):
        raise ContractError(
            "registry_changed", "This conversation belongs to a different source or contract version."
        )


def _question(question):
    if not isinstance(question, str) or not question.strip(JS_TRIM):
        raise ContractError("invalid_question", "A nonempty question is required.")


def applyModelOutput(registry, session, question, rawOutput, expectedExchangeCount=None):
    _current(registry, session)
    if expectedExchangeCount is not None and expectedExchangeCount != len(session["exchanges"]):
        raise ContractError("stale_generation", "The conversation changed during generation.")
    _question(question)
    if not isinstance(rawOutput, str):
        raise ContractError("invalid_model_output", "Model output must be text.")
    state = session["semantic"]
    try:
        model = decodeCalls(rawOutput)
        exchanges = recoveryExchanges(session)
        internal = compileModelCalls(registry, state, model, exchanges)
        result = executeTurn(registry, state, question, internal, exchanges)
        if not result["ok"]:
            err = result["error"]
            raise ContractError(err["code"], err["message"], err["details"])
    except ContractError as e:
        message = e.details.get("clarification")
        return _append(
            session,
            state,
            dict(
                question=question,
                assistant=message if isinstance(message, str) else _FAILURE,
                outcome="error",
                error_code=e.code,
                raw_output=rawOutput,
            ),
        )
    turn = result["turn"]
    return _append(
        session,
        result["state"],
        dict(
            question=question,
            assistant=turn["text"],
            outcome="committed",
            semantic_turn_id=turn["id"],
            raw_output=rawOutput,
            model_calls=model,
            internal_calls=internal,
        ),
    )


def recordProviderFailure(registry, session, question, code="inference_failed"):
    _current(registry, session)
    _question(question)
    if not isinstance(code, str) or not re.fullmatch("[a-z_]+", code):
        raise ContractError("invalid_error_code", "Invalid provider failure classification.")
    return _append(
        session,
        session["semantic"],
        dict(
            question=question,
            assistant=_CONTEXT_LIMIT if code == "context_overflow" else _FAILURE,
            outcome="provider_error",
            error_code=code,
        ),
    )


def recordIncompleteModelOutput(
    registry, session, question, rawOutput, generatedTokenIds, stopReason="token_limit"
):
    _current(registry, session)
    _question(question)
    checked = IncompleteModelOutput(rawOutput, generatedTokenIds, stopReason)
    return _append(
        session,
        session["semantic"],
        dict(
            question=question,
            assistant=_FAILURE,
            outcome="error",
            error_code=checked.code,
            raw_output=rawOutput,
            generation=dict(complete_native_batch=False, token_ids=generatedTokenIds, stop_reason=stopReason),
        ),
    )


def sessionHistory(session):
    semantic = modelHistory(session["semantic"])
    index = {t["id"]: t for t in semantic["turns"]}
    exchanges = []
    for ex in session["exchanges"]:
        row = dict(id=ex["id"], user=ex["question"], assistant=ex["assistant"], outcome=ex["outcome"])
        turn = index.get(ex.get("semantic_turn_id"))
        if turn is not None:
            row.update(
                turn_id=turn["id"],
                calls=turn["calls"],
                actions=turn["actions"],
                clarifications=turn["clarifications"],
            )
        exchanges.append(row)
    return clone(
        dict(
            session_id=session["session_id"],
            exchanges=exchanges,
            pending=semantic["pending"],
            focus=semantic["focus"],
        )
    )


class _SessionKey:
    __slots__ = ("__authenticate",)

    def __init__(self):
        key = secrets.token_bytes(64)
        self.__authenticate = lambda payload: hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()

    def sign(self, payload):
        return self.__authenticate(payload)

    def __reduce__(self):
        raise TypeError("Session keys cannot be serialized.")


def createSessionKey():
    return _SessionKey()


def _assert_key(key):
    if type(key) is not _SessionKey:
        raise ContractError("invalid_persistence_key", "A non-extractable session HMAC key is required.")


def sealSession(session, key):
    _assert_key(key)
    payload = canonical(session)
    return dict(version=1, payload=payload, signature=key.sign(payload))


def restoreSession(registry, envelope, key):
    _assert_key(key)
    if (
        not isinstance(envelope, dict)
        or isinstance(envelope.get("version"), bool)
        or envelope.get("version") != 1
        or not isinstance(envelope.get("payload"), str)
        or not isinstance(envelope.get("signature"), str)
        or not re.fullmatch("[0-9a-f]{64}", envelope["signature"])
    ):
        raise ContractError("invalid_snapshot", "Invalid conversation snapshot.")
    if not hmac.compare_digest(key.sign(envelope["payload"]), envelope["signature"]):
        raise ContractError("snapshot_integrity", "Conversation integrity verification failed.")
    try:
        saved = json.loads(envelope["payload"])
        canonical(saved)
    except (TypeError, ValueError):
        raise ContractError("invalid_snapshot", "Invalid conversation data.") from None
    _current(registry, saved)
    if not isinstance(saved.get("session_id"), str) or not isinstance(saved.get("exchanges"), list):
        raise ContractError("invalid_snapshot", "Missing conversation identity or exchanges.")
    replay = _initial(saved["registry_id"], saved["session_id"], saved.get("dialogue_version"))
    for ex in saved["exchanges"]:
        if ex.get("outcome") == "provider_error":
            replay = recordProviderFailure(registry, replay, ex["question"], ex["error_code"])
        elif ex.get("error_code") == "incomplete_model_output":
            gen = ex.get("generation", {})
            replay = recordIncompleteModelOutput(
                registry,
                replay,
                ex["question"],
                ex.get("raw_output"),
                gen.get("token_ids"),
                gen.get("stop_reason", "token_limit"),
            )
        else:
            replay = applyModelOutput(registry, replay, ex["question"], ex.get("raw_output"))
    if canonical(replay) != canonical(saved):
        raise ContractError(
            "snapshot_replay_mismatch",
            "The stored conversation does not match its actual calls, answers and state.",
        )
    return replay


class _Signal(threading.Event):
    @property
    def aborted(self):
        return self.is_set()


class SessionController:
    def __init__(self, registry, session):
        self.registry = registry
        self.session = session
        self.generation = None
        self._lock = threading.RLock()

    def send(self, question, provider):
        with self._lock:
            if self.generation is not None:
                raise ContractError("generation_in_progress", "A response is already being generated.")
            base = self.session
            signal = _Signal()
            self.generation = signal
        try:
            try:
                raw = provider(
                    dict(
                        question=question,
                        session=base,
                        history=sessionHistory(base),
                        state=base["semantic"],
                        tools=modelTools(self.registry, base["semantic"], recoveryExchanges(base)),
                        signal=signal,
                    )
                )
                result = applyModelOutput(self.registry, base, question, raw)
            except Exception as e:
                if signal.aborted:
                    raise
                if isinstance(e, IncompleteModelOutput):
                    result = recordIncompleteModelOutput(
                        self.registry,
                        base,
                        question,
                        e.details["raw_output"],
                        e.details["generated_token_ids"],
                        e.details["stop_reason"],
                    )
                else:
                    code = (
                        "context_overflow"
                        if isinstance(e, ContractError) and e.code == "context_overflow"
                        else "inference_failed"
                    )
                    result = recordProviderFailure(self.registry, base, question, code)
            with self._lock:
                if self.generation is not signal or self.session is not base:
                    raise ContractError("stale_generation", "The result belongs to an earlier conversation.")
                self.session = result
                return result
        finally:
            with self._lock:
                if self.generation is signal:
                    self.generation = None

    def reset(self):
        with self._lock:
            if self.generation is not None:
                self.generation.set()
            self.generation = None
            self.session = createSession(
                self.registry, dict(dialogueVersion=self.session.get("dialogue_version", 1))
            )
            return self.session
