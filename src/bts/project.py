"""Creation of editable BTS book-project files."""

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from bts.audio import require_mp3_encoder
from bts.constants import AUDIO_FORMATS
from bts.device import resolve_device
from bts.dictionary import PronunciationDictionary, load_dictionary
from bts.epub import get_section_body, load_sections
from bts.errors import GpuMemoryError, TTSError
from bts.languages import SUPPORTED_LANGUAGES, validate_voice
from bts.numbers import prepare_text
from bts.synthesis import synthesize_epub_section_to_file
from bts.textio import read_text_file

DEFAULT_FILENAME_PATTERN = "{index:02d} - {name}"
PROJECT_HEADER = """\
# BTS project file.
# Edit this file to configure the book generated from SourceFile.
# Use "default" for Language, Voice, or OutputFormat to let BTS choose.
# Use {index} and {name} in FilenamePattern for the output index and name.
# Run `bts-tts --build <project.bts>` to build the configured audio files.
# Elements are processed in order:
# - A source element refers to an item in the source; for EPUB files, its id
#   and name come from the section list. A text source has one source element.
#   "default" as id uses that element's 1-based position in the source.
# - A custom element is not in the source, has no id, and must provide name
#   and text.
# - The only supported action is "skip", which omits that element.
# - speech_before and speech_after are optional text spoken around an element.
# - Empty required fields warn and skip that element.
#   "default" uses a field's default value when one is available.
"""


class _ProjectDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


@dataclass(frozen=True)
class _BuildElement:
    name: str
    title_text: str
    body_text: str


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _is_default(value: object) -> bool:
    return isinstance(value, str) and value.strip().casefold() == "default"


def _read_setting(
    config: dict, key: str, default: str | None
) -> tuple[str | None, bool]:
    """Return a project setting and whether it is explicitly configured."""
    if key not in config or _is_default(config[key]):
        return default, False
    value = config[key]
    if not isinstance(value, str):
        raise TTSError(f"Project field '{key}' must be a string.")
    value = value.strip()
    if not value:
        _warn(
            f"Project field '{key}' is empty; using the default value."
        )
        return default, False
    return value, True


def _path_with_extension(path: str, extension: str) -> str:
    result = Path(path)
    if not result.suffix:
        result = Path(f"{result}{extension}")
    return str(result)


def resolve_project_path(name: str) -> Path:
    """Resolve a project argument and append .bts if it has no extension."""
    if not name or not name.strip():
        raise TTSError("A project file name is required.")
    path = Path(name)
    if not path.suffix:
        path = Path(f"{path}.bts")
    return path


def load_project(project_name: str) -> tuple[Path, dict]:
    """Load and validate a BTS project YAML file."""
    project_path = resolve_project_path(project_name)
    try:
        with project_path.open(encoding="utf-8") as project_file:
            data = yaml.safe_load(project_file)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TTSError(
            f"Cannot read project file '{project_path}': {exc}"
        ) from exc

    if not isinstance(data, dict) or not isinstance(
        data.get("BTSProject"), dict
    ):
        raise TTSError(
            f"Project file '{project_path}' must contain a BTSProject mapping."
        )
    return project_path, data["BTSProject"]


def _element_string(
    element: dict, field: str, position: int
) -> str | None:
    value = element.get(field)
    if not isinstance(value, str) or not value.strip():
        _warn(
            f"Project element {position} has an empty required field "
            f"'{field}'; skipping it."
        )
        return None
    return value.strip()


def _optional_speech(element: dict, field: str, position: int) -> str:
    value = element.get(field, "")
    if value is None or _is_default(value):
        return ""
    if not isinstance(value, str):
        raise TTSError(
            f"Project element {position} field '{field}' must be text."
        )
    return value.strip()


def _section_id(value: object, position: int, is_epub: bool) -> int | None:
    if _is_default(value):
        return position if is_epub else 1
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _resolve_elements(
    config: dict, source_file: str
) -> list[_BuildElement]:
    raw_elements = config.get("Elements")
    if not isinstance(raw_elements, list):
        raise TTSError("Project field 'Elements' must be a list.")

    source_path = Path(source_file)
    is_epub = source_path.suffix.casefold() == ".epub"
    source_requests = []
    custom_elements = []

    for position, element in enumerate(raw_elements, 1):
        if not isinstance(element, dict):
            _warn(
                f"Project element {position} is not a mapping; skipping it."
            )
            continue

        action = element.get("action", "")
        if action is None or _is_default(action):
            action = ""
        if action:
            if not isinstance(action, str):
                raise TTSError(
                    f"Project element {position} has an invalid action."
                )
            if action.casefold() == "skip":
                continue
            raise TTSError(
                f"Project element {position} has unsupported action "
                f"'{action}'."
            )

        element_type = element.get("type")
        if _is_default(element_type):
            element_type = "source"
        if not isinstance(element_type, str) or not element_type.strip():
            _warn(
                f"Project element {position} has an empty required field "
                "'type'; skipping it."
            )
            continue
        element_type = element_type.strip().casefold()
        if element_type not in {"source", "custom"}:
            raise TTSError(
                f"Project element {position} has unsupported type "
                f"'{element_type}'."
            )

        name_value = element.get("name")
        use_default_name = _is_default(name_value)
        if not use_default_name:
            name_value = _element_string(element, "name", position)
            if name_value is None:
                continue

        speech_before = _optional_speech(
            element, "speech_before", position
        )
        speech_after = _optional_speech(element, "speech_after", position)

        if element_type == "source":
            raw_id = element.get("id")
            if raw_id is None or (
                isinstance(raw_id, str) and not raw_id.strip()
            ):
                _warn(
                    f"Project element {position} has an empty required "
                    "field 'id'; skipping it."
                )
                continue
            source_id = _section_id(raw_id, position, is_epub)
            source_requests.append(
                (
                    position,
                    source_id,
                    name_value if not use_default_name else None,
                    speech_before,
                    speech_after,
                )
            )
            continue

        raw_text = element.get("text")
        if (
            not isinstance(raw_text, str)
            or not raw_text.strip()
            or _is_default(raw_text)
        ):
            _warn(
                f"Project element {position} has no default for required "
                "field 'text'; skipping it."
            )
            continue
        name = (
            f"Custom {position}"
            if use_default_name
            else name_value
        )
        title_text = speech_before
        body_text = "\n".join(
            part for part in (raw_text.strip(), speech_after) if part
        )
        custom_elements.append(
            (position, _BuildElement(name, title_text, body_text))
        )

    resolved_sources = []
    if source_requests:
        if is_epub:
            sections = load_sections(source_file)
            sections_by_id = {section.index: section for section in sections}
            for position, source_id, name, before, after in source_requests:
                section = sections_by_id.get(source_id)
                if section is None:
                    continue
                body = get_section_body(section)
                if (
                    not body
                    and not section.has_heading
                    and not before
                    and not after
                ):
                    _warn(
                        f"Project element {position} refers to an empty "
                        "section; skipping it."
                    )
                    continue
                if name is None:
                    name = section.title
                title_text = "\n".join(
                    part
                    for part in (
                        before,
                        section.title if section.has_heading else "",
                    )
                    if part
                )
                body_text = "\n".join(
                    part for part in (body, after) if part
                )
                resolved_sources.append(
                    (position, _BuildElement(name, title_text, body_text))
                )
        else:
            text_requests = [
                request for request in source_requests
                if request[1] == 1
            ]
            if text_requests:
                source_text = read_text_file(source_file)
                for position, _, name, before, after in text_requests:
                    if name is None:
                        name = source_path.stem
                    body_text = "\n".join(
                        part for part in (source_text, after) if part
                    )
                    resolved_sources.append(
                        (
                            position,
                            _BuildElement(name, before, body_text),
                        )
                    )

    return [
        item
        for _, item in sorted(resolved_sources + custom_elements)
    ]


def _safe_name(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name or "section"


def _output_filename(
    pattern: str, index: int, name: str, audio_format: str
) -> str:
    try:
        filename = pattern.format(index=index, name=_safe_name(name))
    except (IndexError, KeyError, ValueError, TypeError) as exc:
        raise TTSError(
            f"Invalid FilenamePattern '{pattern}': {exc}"
        ) from exc
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise TTSError(
            "FilenamePattern must produce a simple file name without "
            "directory separators."
        )
    return f"{filename}.{audio_format}"


def _override_value(
    key: str,
    project_value: str | None,
    is_configured: bool,
    override: str | None,
) -> str | None:
    if override is None:
        return project_value
    if is_configured and override.casefold() != str(project_value).casefold():
        _warn(
            f"Command-line value for {key} overrides the project value "
            f"'{project_value}'."
        )
    return override


def create_project(
    source_file: str,
    project_name: str | None = None,
    source_type: Literal["epub", "text"] = "epub",
) -> str:
    """Create a BTS project YAML file for an EPUB or text-file source."""
    source_path = Path(source_file)
    title = source_path.stem

    if source_type == "epub":
        sections = load_sections(source_file)
        elements = [
            {"type": "source", "id": section.index, "name": section.title}
            for section in sections
        ]
    else:
        read_text_file(source_file)
        elements = [{"type": "source", "id": 1, "name": title}]

    project_path = (
        resolve_project_path(project_name)
        if project_name
        else source_path.with_suffix(".bts")
    )

    data = {
        "BTSProject": {
            "Title": title,
            "SourceFile": source_file,
            "Dictionary": str(source_path.with_suffix(".dict")),
            "Language": "default",
            "Voice": "default",
            "OutputFormat": "default",
            "FilenamePattern": DEFAULT_FILENAME_PATTERN,
            "Elements": elements,
        }
    }

    try:
        with project_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as project_file:
            project_file.write(PROJECT_HEADER)
            yaml.dump(
                data,
                project_file,
                Dumper=_ProjectDumper,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
                indent=2,
            )
    except FileExistsError as exc:
        raise TTSError(
            f"Project file '{project_path}' already exists."
        ) from exc
    except OSError as exc:
        raise TTSError(
            f"Cannot write project file '{project_path}': {exc}"
        ) from exc

    return str(project_path)


def build_project(
    project_name: str,
    *,
    output_dir: str | None = None,
    language_override: str | None = None,
    voice_override: str | None = None,
    format_override: str | None = None,
    dictionary_override: str | None = None,
    dictionary_ignore_case: bool = False,
    speed: float = 1.0,
    device: str = "auto",
    mp3_quality: int = 4,
    max_pause_seconds: float = 0.5,
) -> int:
    """Build one audio file per usable element in a BTS project."""
    _, config = load_project(project_name)
    source_file = config.get("SourceFile")
    if (
        not isinstance(source_file, str)
        or not source_file.strip()
        or _is_default(source_file)
    ):
        raise TTSError("Project field 'SourceFile' is required.")

    elements = _resolve_elements(config, source_file)
    if not elements:
        print("[BTS Project] No buildable elements.")
        return 0

    project_language, language_is_configured = _read_setting(
        config, "Language", "es"
    )
    language = _override_value(
        "--lang",
        project_language,
        language_is_configured,
        language_override,
    )
    language = (language or "es").lower()
    if language not in SUPPORTED_LANGUAGES:
        raise TTSError(f"Unsupported project language '{language}'.")

    project_voice, voice_is_configured = _read_setting(
        config, "Voice", None
    )
    voice = _override_value(
        "--voice", project_voice, voice_is_configured, voice_override
    )

    project_format, format_is_configured = _read_setting(
        config, "OutputFormat", "flac"
    )
    audio_format = _override_value(
        "--format",
        project_format,
        format_is_configured,
        format_override,
    )
    audio_format = (audio_format or "flac").lower()
    if audio_format not in AUDIO_FORMATS:
        raise TTSError(
            f"Unsupported project output format '{audio_format}'."
        )

    source_path = Path(source_file)
    default_dictionary = str(source_path.with_suffix(".dict"))
    project_dictionary, dictionary_is_configured = _read_setting(
        config, "Dictionary", default_dictionary
    )
    if project_dictionary:
        project_dictionary = _path_with_extension(
            project_dictionary, ".dict"
        )
    dictionary = (
        _path_with_extension(dictionary_override, ".dict")
        if dictionary_override is not None and dictionary_override
        else dictionary_override
    )
    if dictionary_override is not None:
        if (
            dictionary_is_configured
            and dictionary
            and dictionary != project_dictionary
        ):
            _warn(
                "Command-line value for --dictionary overrides the "
                f"project value '{project_dictionary}'."
            )
    else:
        dictionary = project_dictionary

    filename_pattern, _ = _read_setting(
        config, "FilenamePattern", DEFAULT_FILENAME_PATTERN
    )
    filename_pattern = filename_pattern or DEFAULT_FILENAME_PATTERN
    default_output_dir = source_path.stem
    output_dir = output_dir or default_output_dir

    output_paths = [
        os.path.join(
            output_dir,
            _output_filename(
                filename_pattern, index, element.name, audio_format
            ),
        )
        for index, element in enumerate(elements, 1)
    ]
    pending = []
    for element, output_path in zip(elements, output_paths, strict=True):
        if os.path.isfile(output_path):
            print(
                f"[BTS Project] Skipping existing file '{output_path}'."
            )
        else:
            pending.append((element, output_path))
    if not pending:
        print("[BTS Project] All project elements are already built.")
        return 0

    if dictionary:
        if os.path.isfile(dictionary):
            lexicon = load_dictionary(dictionary, dictionary_ignore_case)
            print(
                f"[INFO] Loaded {len(lexicon)} pronunciation entries from "
                f"'{dictionary}'."
            )
        else:
            _warn(
                f"Dictionary '{dictionary}' does not exist; continuing "
                "without it."
            )
            lexicon = PronunciationDictionary({})
    else:
        lexicon = PronunciationDictionary({})

    validate_voice(language, voice)
    if audio_format == "mp3":
        require_mp3_encoder()
    resolved_device = resolve_device(device)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise TTSError(
            f"Cannot create output directory '{output_dir}': {exc}"
        ) from exc

    failed = 0
    print(
        f"[BTS Project] Building {len(pending)} element(s) into "
        f"'{output_dir}'."
    )
    for element, output_path in pending:
        try:
            synthesize_epub_section_to_file(
                prepare_text(element.title_text, language, lexicon),
                prepare_text(element.body_text, language, lexicon),
                output_path,
                lang_code=language,
                voice=voice,
                speed=speed,
                device=resolved_device,
                audio_format=audio_format,
                mp3_quality=mp3_quality,
                max_pause_seconds=max_pause_seconds,
            )
        except GpuMemoryError:
            raise
        except TTSError as exc:
            failed += 1
            _warn(f"Failed to build '{output_path}': {exc}")

    print(
        f"[BTS Project] Finished: {len(pending) - failed} built, "
        f"{failed} failed."
    )
    return 1 if failed else 0
