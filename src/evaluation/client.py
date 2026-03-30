"""Clients for invoking a System Under Test (SUT) and collecting its responses.

Provides subprocess-based clients (``ScriptSUTClient``, ``ScriptSUTMCPClient``)
that launch an external script, feed it user questions, and parse the structured
output, as well as ``OpenSourceClient`` which calls an OpenAI-compatible API
directly.  All clients implement the ``BaseSUTClient`` interface so the
evaluation runner can treat them interchangeably.
"""

import ast
import json
import os
import re
import subprocess
import threading
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, ClassVar, Optional

from langsmith.wrappers import wrap_openai
from openai import OpenAI

from src.common.config import ClientConfig


class PromptTimeoutError(Exception):
    """Raised when the script does not print the expected prompt within the configured timeout."""


_EMPTY_RESPONSE: dict[str, Any] = {
    "answer": "",
    "agent_name": "",
    "thinking": "",
    "report_agent": {},
}


class BaseSUTClient(ABC):
    """Abstract base for all SUT clients."""

    @abstractmethod
    def predict(self, inputs: dict, **kwargs: Any) -> dict:
        """Send *inputs* to the SUT and return its structured response."""


class ScriptSUTClient(BaseSUTClient):
    """Runs a SUT script as a subprocess and communicates via stdin/stdout."""

    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        script_path: Optional[str] = None,
        cwd: Optional[str] = None,
    ):
        if config:
            self._mode = config.mode
            self._timeout = config.timeout
            # Allow explicit overrides; fall back to config values when not provided.
            self.cwd = cwd if cwd is not None else config.cwd
            self.script_path = (
                script_path if script_path is not None else config.script_path
            )
        else:
            # No config: use same defaults as ClientConfig (project-relative cwd, no machine-specific paths)
            defaults = ClientConfig()
            self._mode = "single_round"
            self._timeout = defaults.timeout
            self.cwd = cwd if cwd is not None else defaults.cwd
            self.script_path = (
                script_path if script_path is not None else defaults.script_path
            )

        self._process: Optional[subprocess.Popen[str]] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _build_env(
        self,
        parent_headers: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env["PYTHONUNBUFFERED"] = "1"
        # Pass LangSmith parent run context so SUT subprocess can attach traces to the evaluation run
        if parent_headers:
            env["LANGSMITH_PARENT_HEADERS"] = json.dumps(parent_headers)
        return env

    def _spawn_process(
        self,
        parent_headers: Optional[dict[str, str]] = None,
    ) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ["uv", "run", self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self.cwd,
            env=self._build_env(parent_headers=parent_headers),
        )

    def _start_stderr_drain(self, proc: subprocess.Popen[str]) -> None:
        """Start a background thread to drain stderr, preventing blocking if the child writes to stderr."""
        assert proc.stderr is not None

        def drain_stderr() -> None:
            try:
                # Read stderr until EOF (process dies) or pipe closes
                while True:
                    chunk = proc.stderr.read(4096)  # type: ignore[union-attr]
                    if not chunk:
                        break
                    # Optionally log stderr for debugging; for now, discard to prevent blocking
            except Exception:
                # Thread exits when process dies or pipe closes
                pass

        self._stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        self._stderr_thread.start()

    def start(self) -> "ScriptSUTClient":
        if self._mode != "multi_round":
            return self
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = self._spawn_process()
                self._start_stderr_drain(self._process)
                try:
                    self._read_until_prompt(self._process)
                except PromptTimeoutError:
                    self._process = None
                    self._stderr_thread = None
                    raise
        return self

    def stop(self) -> None:
        with self._lock:
            if self._process is not None:
                try:
                    self._process.stdin.close()  # type: ignore[union-attr]
                    self._process.wait(timeout=5)
                except Exception:
                    self._process.kill()
                    self._process.wait()
                finally:
                    self._process = None
                    self._stderr_thread = None  # Thread will exit when process dies

    def __enter__(self) -> "ScriptSUTClient":
        return self.start()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        # Ensure that cleanup errors do not mask the original exception from the with-block.
        try:
            self.stop()
        except Exception:
            # Best-effort cleanup; ignore errors here to let any original exception propagate.
            pass

    @staticmethod
    def _extract_braced_block(text: str) -> str:
        start = text.find("{")
        if start == -1:
            return ""
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]

    @staticmethod
    def _parse_literal(raw: str) -> Any:
        if not raw:
            return {}
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw

    @staticmethod
    def _parse_report_agent(stdout: str) -> dict[str, Any]:
        report_matches = list(re.finditer(r"\[ReportAgent\]", stdout))
        if not report_matches:
            return {}

        sql_raw = ""
        data_raw = ""
        for match in report_matches:
            raw = ScriptSUTClient._extract_braced_block(stdout[match.end() :])
            parsed = ScriptSUTClient._parse_literal(raw)
            keys = parsed.keys() if isinstance(parsed, dict) else set()
            if "result" in keys:
                data_raw = raw
            elif "sql" in keys or (isinstance(parsed, str) and "'sql'" in parsed):
                sql_raw = raw

        return {
            "sql_result": ScriptSUTClient._parse_literal(sql_raw),
            "data_result": ScriptSUTClient._parse_literal(data_raw),
        }

    _NOISE_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"^\[DEBUG\] Supervisor graph compiled with MemorySaver checkpointer\."
        ),
        re.compile(r"^Routing human input to orchestrator$"),
        re.compile(r"^Token count for \d+ messages: \d+$"),
        re.compile(r"^Token threshold check: .+$"),
    ]

    @staticmethod
    def _parse_stdout(stdout: str, cwd: Optional[str] = None) -> dict[str, Any]:
        # Normalize SUT logs: keep only the message content after "sut:" on each line
        # so downstream parsing operates on the assistant/supervisor text instead of
        # full timestamped log lines.
        lines = stdout.splitlines()
        stripped_lines: list[str] = []
        for line in lines:
            # Keep only text after "sut:" if present
            if " sut:" in line:
                _, after = line.split(" sut:", 1)
                msg = after.strip()
            else:
                msg = line.strip()

            # Ignore debug/noise lines from the SUT subprocess
            if msg and all(ch == "=" for ch in msg):
                continue
            if any(p.search(msg) for p in ScriptSUTClient._NOISE_PATTERNS):
                continue

            stripped_lines.append(msg)
        normalized_stdout = "\n".join(stripped_lines)

        you_match = re.search(r"\[You\]:.*\n?", normalized_stdout)
        thinking_start = you_match.end() if you_match else 0
        first_marker = re.search(
            r"\[ReportAgent\]|\[AI - [^\]]+\]", normalized_stdout[thinking_start:]
        )
        if first_marker:
            thinking_end = thinking_start + first_marker.start()
        else:
            thinking_end = len(normalized_stdout)
        thinking = normalized_stdout[thinking_start:thinking_end].strip()

        report_agent = ScriptSUTClient._parse_report_agent(normalized_stdout)
        matches = list(re.finditer(r"\[AI - ([^\]]+)\]:?", normalized_stdout))

        if matches:
            # Collect text from each [AI - agent] block
            agent_blocks: list[tuple[str, str]] = []
            for i, match in enumerate(matches):
                start = match.end()
                end = (
                    matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(normalized_stdout)
                )
                block_text = normalized_stdout[start:end].strip()
                agent_blocks.append((match.group(1).strip(), block_text))

            # Use the first non-orchestrator agent name if available
            agent_name = agent_blocks[-1][0]
            for name, _ in agent_blocks:
                if name != "orchestrator":
                    agent_name = name
                    break

            # Concatenate all block texts
            raw_answer = "\n\n".join(text for _, text in agent_blocks if text)

            routing_msg = "Routing human input to orchestrator"
            if raw_answer.endswith(routing_msg):
                raw_answer = raw_answer[: -len(routing_msg)].strip()

            if "[You]:" in raw_answer:
                raw_answer = raw_answer.rsplit("[You]:", 1)[0].strip()
                # Detect which agent produced the answer and read appropriate file
                folder_path = ""
                file_name = ""
                if agent_name == "subscription_agent":
                    folder_path = "saved_subscriptions"
                    file_name = "subscription.json"
                elif agent_name == "bundle_agent":
                    folder_path = "saved_bundles"
                    file_name = "bundle.json"
                elif agent_name == "promotions_agent":
                    folder_path = "saved_promotions"
                    file_name = "promotion.json"

                file_path = os.path.join(cwd or ".", folder_path, file_name)
                # Read the JSON file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_answer = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    return {
                        "answer": f"Error reading file {file_path}: {str(e)}\nRaw response: {raw_answer}",
                        "agent_name": agent_name,
                        "thinking": thinking,
                        "report_agent": report_agent,
                    }
            return {
                "answer": raw_answer,
                "agent_name": agent_name,
                "thinking": thinking,
                "report_agent": report_agent,
            }

        return {
            "answer": normalized_stdout.strip(),
            "agent_name": "",
            "thinking": thinking,
            "report_agent": report_agent,
        }

    def _predict_single_round(
        self,
        question: str | list[str],
        parent_headers: Optional[dict[str, str]] = None,
    ) -> dict:
        if isinstance(question, list):
            # baseline chat data is a list of messages
            # we only need the first message
            if not question:
                return {**_EMPTY_RESPONSE, "answer": "Error: Empty question"}
            question = question[0]
        question = " ".join(question.split("\n")).strip()

        try:
            process = self._spawn_process(parent_headers=parent_headers)
            stdout, _ = process.communicate(
                input=question + "\n", timeout=self._timeout
            )
            return self._parse_stdout(stdout, self.cwd)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return {**_EMPTY_RESPONSE, "answer": "Error: Timeout"}
        except Exception as e:
            return {**_EMPTY_RESPONSE, "answer": f"Error: {e}"}

    def _read_until_prompt(self, proc: subprocess.Popen[str]) -> str:
        """Read char-by-char until '[You]:' appears (prompt has no trailing newline).
        Raises PromptTimeoutError if the prompt is not seen within self._timeout seconds;
        the process is terminated when a timeout occurs.
        """
        assert proc.stdout is not None
        sentinel = "[You]:"
        result_holder: list[str | None] = [None]

        def read_loop() -> None:
            buf: list[str] = []
            try:
                while True:
                    ch = proc.stdout.read(1)  # type: ignore[union-attr]
                    if not ch:
                        break
                    buf.append(ch)
                    if (
                        len(buf) >= len(sentinel)
                        and "".join(buf[-len(sentinel) :]) == sentinel
                    ):
                        break
            finally:
                result_holder[0] = "".join(buf)

        reader = threading.Thread(target=read_loop, daemon=True)
        reader.start()
        reader.join(timeout=self._timeout)
        if reader.is_alive():
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
            raise PromptTimeoutError(
                f"Script did not print '[You]:' within {self._timeout}s. "
                "Process terminated."
            )
        assert result_holder[0] is not None
        return result_holder[0]

    def _predict_multi_round(
        self,
        question: str,
        parent_headers: Optional[dict[str, str]] = None,
    ) -> dict:
        with self._lock:
            proc = self._process
            if proc is None or proc.poll() is not None:
                proc = self._spawn_process()
                self._process = proc
                self._start_stderr_drain(proc)
                try:
                    self._read_until_prompt(proc)
                except Exception as e:
                    # Clean up on failure during initial prompt read and return structured error.
                    try:
                        proc.kill()
                        proc.wait()
                    except Exception:
                        pass
                    if self._process is proc:
                        self._process = None
                        self._stderr_thread = None
                    return {**_EMPTY_RESPONSE, "answer": f"Error: {e}"}

        try:
            assert proc.stdin is not None
            # Hold lock for entire write->read so concurrent predict() calls don't interleave stdin/stdout
            with self._lock:
                # multi_round protocol: optional control line (parent headers for this request), then question
                control = json.dumps(
                    {"langsmith_parent_headers": parent_headers}
                    if parent_headers
                    else {}
                )
                proc.stdin.write(control + "\n")
                question = " ".join(question.split("\n")).strip()
                proc.stdin.write(question + "\n")
                proc.stdin.flush()
                raw = self._read_until_prompt(proc)

            sentinel = "[You]:"
            if raw.endswith(sentinel):
                raw = raw[: -len(sentinel)]

            return self._parse_stdout(raw)
        except Exception as e:
            with self._lock:
                if self._process is proc:
                    try:
                        proc.kill()
                        proc.wait()
                    except Exception:
                        pass
                    self._process = None
                    self._stderr_thread = None
            return {**_EMPTY_RESPONSE, "answer": f"Error: {e}"}

    def predict(self, inputs: dict, **kwargs: Any) -> dict:
        question = inputs.get("question", "")
        if not question:
            return dict(_EMPTY_RESPONSE)

        parent_headers = kwargs.get("langsmith_parent_headers")
        if self._mode == "multi_round":
            return self._predict_multi_round(question, parent_headers=parent_headers)
        return self._predict_single_round(question, parent_headers=parent_headers)


class ScriptSUTMCPClient(BaseSUTClient):
    """Runs an MCP-based SUT agent script as a subprocess."""

    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        script_path: Optional[str] = None,
        cwd: Optional[str] = None,
    ):
        if config:
            self._mode = config.mode
            self._timeout = config.timeout
            # Allow explicit overrides; fall back to config values when not provided.
            self.cwd = cwd if cwd is not None else config.cwd
            self.script_path = (
                script_path if script_path is not None else config.script_path
            )
        else:
            # No config: use same defaults as ClientConfig (project-relative cwd, no machine-specific paths)
            defaults = ClientConfig()
            self._mode = "single_round"
            self._timeout = defaults.timeout
            self.cwd = cwd if cwd is not None else defaults.cwd
            self.script_path = (
                script_path if script_path is not None else defaults.script_path
            )

        self._process: Optional[subprocess.Popen[str]] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _build_env(
        self,
        parent_headers: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env["PYTHONUNBUFFERED"] = "1"
        # Pass LangSmith parent run context so SUT subprocess can attach traces to the evaluation run
        if parent_headers:
            env["LANGSMITH_PARENT_HEADERS"] = json.dumps(parent_headers)
        return env

    def _spawn_process(
        self,
        parent_headers: Optional[dict[str, str]] = None,
    ) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ["uv", "run", self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self.cwd,
            env=self._build_env(parent_headers=parent_headers),
        )

    def _start_stderr_drain(self, proc: subprocess.Popen[str]) -> None:
        """Start a background thread to drain stderr, preventing blocking if the child writes to stderr."""
        assert proc.stderr is not None

        def drain_stderr() -> None:
            try:
                # Read stderr until EOF (process dies) or pipe closes
                while True:
                    chunk = proc.stderr.read(4096)  # type: ignore[union-attr]
                    if not chunk:
                        break
                    # Optionally log stderr for debugging; for now, discard to prevent blocking
            except Exception:
                # Thread exits when process dies or pipe closes
                pass

        self._stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        self._stderr_thread.start()

    def start(self) -> "ScriptSUTMCPClient":
        if self._mode != "multi_round":
            return self
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = self._spawn_process()
                self._start_stderr_drain(self._process)
                try:
                    self._read_until_prompt(self._process)
                except PromptTimeoutError:
                    self._process = None
                    self._stderr_thread = None
                    raise
        return self

    def _read_until_prompt(self, proc: subprocess.Popen[str]) -> str:
        """Read char-by-char until '[You]:' appears (prompt has no trailing newline).
        Raises PromptTimeoutError if the prompt is not seen within self._timeout seconds;
        the process is terminated when a timeout occurs.
        """
        assert proc.stdout is not None
        sentinel = "[You]:"
        result_holder: list[str | None] = [None]

        def read_loop() -> None:
            buf: list[str] = []
            try:
                while True:
                    ch = proc.stdout.read(1)  # type: ignore[union-attr]
                    if not ch:
                        break
                    buf.append(ch)
                    if (
                        len(buf) >= len(sentinel)
                        and "".join(buf[-len(sentinel) :]) == sentinel
                    ):
                        break
            finally:
                result_holder[0] = "".join(buf)

        reader = threading.Thread(target=read_loop, daemon=True)
        reader.start()
        reader.join(timeout=self._timeout)
        if reader.is_alive():
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
            raise PromptTimeoutError(
                f"Script did not print '[You]:' within {self._timeout}s. "
                "Process terminated."
            )
        assert result_holder[0] is not None
        return result_holder[0]

    def stop(self) -> None:
        with self._lock:
            if self._process is not None:
                try:
                    self._process.stdin.close()  # type: ignore[union-attr]
                    self._process.wait(timeout=5)
                except Exception:
                    self._process.kill()
                    self._process.wait()
                finally:
                    self._process = None
                    self._stderr_thread = None  # Thread will exit when process dies

    def __enter__(self) -> "ScriptSUTMCPClient":
        return self.start()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        # Ensure that cleanup errors do not mask the original exception from the with-block.
        try:
            self.stop()
        except Exception:
            # Best-effort cleanup; ignore errors here to let any original exception propagate.
            pass

    def predict(self, inputs: dict, **kwargs: Any) -> dict:
        """
        Run the SUT agent script with the given question.
        Returns dictionary containing "answer" and "agent_name".
        """

        question = inputs.get("question", "")

        if not question:
            return {"answer": "", "agent_name": ""}

        question = " ".join(question.split("\n")).strip()
        proc_inputs = [question]
        input_str = "\n".join(proc_inputs) + "\n"

        # Construct command
        command = [
            "uv",
            "run",
            "-m",
            "src.sut.agents.mcp_agent.mcp_based_creation_agent",
        ]  # Using module path to ensure correct imports

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.cwd,
                env=self._build_env(
                    parent_headers=kwargs.get("langsmith_parent_headers")
                ),
            )

            stdout, stderr = process.communicate(input=input_str, timeout=self._timeout)

            # Parse output
            matches = list(re.finditer(r"\[AI\]:\s*", stdout))

            if matches:
                last_match = matches[-1]

                agent_name = "mcp_agent"

                start_index = last_match.end()

                raw_answer = stdout[start_index:].strip()

                # Extract file path from response.
                # Supports:
                # 1) absolute Windows path (C:\...\file.json)
                # 2) relative path or bare filename (file.json)
                file_path_match = re.search(
                    r'([A-Za-z]:[/\\](?:[^/\\:*?"<>|\r\n]+[/\\])*[^/\\:*?"<>|\r\n]+\.json)',
                    raw_answer,
                )
                if not file_path_match:
                    file_path_match = re.search(
                        r'(?<![A-Za-z]:[/\\])([^\s"\'<>:|?*]+\.json)',
                        raw_answer,
                    )

                if file_path_match:
                    file_path = file_path_match.group(1)
                    if not os.path.isabs(file_path):
                        file_path = os.path.join(self.cwd or ".", file_path)

                    try:
                        # Read the JSON file
                        with open(file_path, "r", encoding="utf-8") as f:
                            json_content = json.load(f)

                        # Return the JSON content as the answer
                        return {
                            "answer": json.dumps(json_content),
                            "agent_name": agent_name,
                        }
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        # If file reading fails, return the raw answer with error info
                        return {
                            "answer": f"Error reading file {file_path}: {str(e)}\nRaw response: {raw_answer}",
                            "agent_name": agent_name,
                        }

                # If no file path found, return the raw answer
                return {"answer": raw_answer, "agent_name": agent_name}

            return {"answer": stdout.strip(), "agent_name": ""}

        except subprocess.TimeoutExpired:
            if process:
                process.kill()
            return {"answer": "Error: Timeout", "agent_name": ""}
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "agent_name": ""}


class OpenSourceClient(BaseSUTClient):
    """SUT client that queries an OpenAI-compatible model directly."""

    def __init__(self, model_name: str = "gpt-5-mini"):
        self.model_name = model_name
        self.client = wrap_openai(OpenAI())

    def predict(self, inputs: dict, **kwargs: Any) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the following question accurately in JSON format. "
                            "The JSON should contain two keys: 'answer' (the response to the question) "
                            "and 'agent_name'. The 'agent_name' must be one of the following: "
                            "'customer_support_agent', 'none_sut_agent', 'promotions_agent', 'report_agent'."
                        ),
                    },
                    {"role": "user", "content": inputs["question"]},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content is None:
                return {"answer": "", "agent_name": "opensource_agent"}

            try:
                data = json.loads(content)
                return {
                    "answer": data.get("answer", "").strip(),
                    "agent_name": data.get("agent_name", "opensource_agent"),
                }
            except json.JSONDecodeError:
                return {
                    "answer": content.strip(),
                    "agent_name": "opensource_agent",
                }

        except Exception as e:
            return {"answer": f"Error: {str(e)}", "agent_name": "error"}
