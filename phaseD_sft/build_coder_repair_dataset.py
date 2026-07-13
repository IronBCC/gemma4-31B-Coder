#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseD_sft.agentic_trace_filters import trace_should_keep


@dataclass(frozen=True)
class RepairCase:
    name: str
    prompt: str
    answer: str


REPAIR_CASES: tuple[RepairCase, ...] = (
    RepairCase(
        name="lru_cache_mutable_list_recent_files",
        prompt="""patch only

src/recent.py
from functools import lru_cache

@lru_cache
def default_recent_files():
    return []

failed:
files = default_recent_files()
files.append("notes.txt")
assert default_recent_files() == []
assert isinstance(default_recent_files(), list)

The API must keep returning a list, but callers must not share cached mutable state.
""",
        answer="""--- a/src/recent.py
+++ b/src/recent.py
@@ -1,5 +1,3 @@
-from functools import lru_cache
-
-@lru_cache
 def default_recent_files():
     return []""",
    ),
    RepairCase(
        name="lru_cache_mutable_dict_headers",
        prompt="""patch only

src/headers.py
from functools import lru_cache

@lru_cache
def default_headers():
    return {}

failed:
headers = default_headers()
headers["X-Debug"] = "1"
assert default_headers() == {}

Do not return a cached mutable dict to callers.
""",
        answer="""--- a/src/headers.py
+++ b/src/headers.py
@@ -1,5 +1,3 @@
-from functools import lru_cache
-
-@lru_cache
 def default_headers():
     return {}""",
    ),
    RepairCase(
        name="cache_mutable_set_features",
        prompt="""patch only

src/features.py
from functools import cache

@cache
def enabled_features():
    return set()

failed:
features = enabled_features()
features.add("beta")
assert enabled_features() == set()
assert isinstance(enabled_features(), set)

Keep the set return type. Do not share cached mutable state.
""",
        answer="""--- a/src/features.py
+++ b/src/features.py
@@ -1,5 +1,3 @@
-from functools import cache
-
-@cache
 def enabled_features():
     return set()""",
    ),
    RepairCase(
        name="lru_cache_nested_mutable_template",
        prompt="""patch only

src/template.py
from functools import lru_cache

@lru_cache
def default_template():
    return {"blocks": []}

failed:
template = default_template()
template["blocks"].append("hero")
assert default_template() == {"blocks": []}

Do not use cached mutable nested defaults here.
""",
        answer="""--- a/src/template.py
+++ b/src/template.py
@@ -1,5 +1,3 @@
-from functools import lru_cache
-
-@lru_cache
 def default_template():
     return {"blocks": []}""",
    ),
    RepairCase(
        name="cached_property_mutable_list_permissions",
        prompt="""patch only

src/account.py
from functools import cached_property

class Account:
    @cached_property
    def permissions(self):
        return []

failed:
account = Account()
account.permissions.append("admin")
assert account.permissions == []

Keep attribute-style access as account.permissions, but do not cache a mutable default.
""",
        answer="""--- a/src/account.py
+++ b/src/account.py
@@ -1,7 +1,7 @@
-from functools import cached_property
-
 class Account:
-    @cached_property
+    @property
     def permissions(self):
         return []""",
    ),
    RepairCase(
        name="cached_property_mutable_dict_metadata",
        prompt="""patch only

src/resource.py
from functools import cached_property

class Resource:
    @cached_property
    def metadata(self):
        return {}

failed:
r = Resource()
r.metadata["owner"] = "alice"
assert r.metadata == {}

Keep r.metadata property syntax. Do not return a stored mutable default.
""",
        answer="""--- a/src/resource.py
+++ b/src/resource.py
@@ -1,7 +1,7 @@
-from functools import cached_property
-
 class Resource:
-    @cached_property
+    @property
     def metadata(self):
         return {}""",
    ),
    RepairCase(
        name="cached_property_nested_mutable_profile",
        prompt="""patch only

src/profile.py
from functools import cached_property

class Profile:
    @cached_property
    def defaults(self):
        return {"tags": []}

failed:
p = Profile()
p.defaults["tags"].append("vip")
assert p.defaults == {"tags": []}

Attribute access must stay p.defaults. Avoid cached mutable nested state.
""",
        answer="""--- a/src/profile.py
+++ b/src/profile.py
@@ -1,7 +1,7 @@
-from functools import cached_property
-
 class Profile:
-    @cached_property
+    @property
     def defaults(self):
         return {"tags": []}""",
    ),
    RepairCase(
        name="cached_property_mutable_set_roles",
        prompt="""patch only

src/member.py
from functools import cached_property

class Member:
    @cached_property
    def roles(self):
        return set()

failed:
m = Member()
m.roles.add("owner")
assert m.roles == set()
assert isinstance(m.roles, set)

Keep m.roles attribute access and set return type. Do not cache the mutable set.
""",
        answer="""--- a/src/member.py
+++ b/src/member.py
@@ -1,7 +1,7 @@
-from functools import cached_property
-
 class Member:
-    @cached_property
+    @property
     def roles(self):
         return set()""",
    ),
    RepairCase(
        name="stale_fixture_year_complete",
        prompt="""patch only
today 2026-07-08

src/year.py
def display_year(now):
    return now.year

tests/test_year.py
class FrozenNow:
    year = 2025

def test_display_year():
    assert display_year(FrozenNow()) == 2025

pytest now reports expected 2025 but got 2026. Production logic is correct; the test fixture and expected value are stale.
""",
        answer="""--- a/tests/test_year.py
+++ b/tests/test_year.py
@@ -1,6 +1,6 @@
 class FrozenNow:
-    year = 2025
+    year = 2026
 
 def test_display_year():
-    assert display_year(FrozenNow()) == 2025
+    assert display_year(FrozenNow()) == 2026""",
    ),
    RepairCase(
        name="stale_snapshot_year_complete",
        prompt="""patch only
today 2026-07-08

src/banner.py returns f"Generated {now.year}".
tests/test_banner.py uses FrozenClock.year = 2025 and asserts "Generated 2025".
pytest reports "Generated 2026" != "Generated 2025".
Production logic is right; update the stale test only.
""",
        answer="""--- a/tests/test_banner.py
+++ b/tests/test_banner.py
@@ -1,6 +1,6 @@
 class FrozenClock:
-    year = 2025
+    year = 2026
 
 def test_banner():
-    assert banner(FrozenClock()) == "Generated 2025"
+    assert banner(FrozenClock()) == "Generated 2026\"""",
    ),
    RepairCase(
        name="tempfile_windows_namedtemporaryfile_delete_false",
        prompt="""patch only

src/export.py
from tempfile import NamedTemporaryFile

def pass_file_to_tool(data, tool):
    with NamedTemporaryFile() as f:
        f.write(data)
        f.flush()
        return tool(f.name)

failed on Windows: the external tool cannot reopen the temp file while it is open.
Create a named temp file that can be reopened, close it before calling the tool, and remove it afterward without hiding tool exceptions.
""",
        answer="""--- a/src/export.py
+++ b/src/export.py
@@ -1,7 +1,15 @@
+import os
 from tempfile import NamedTemporaryFile
 
 def pass_file_to_tool(data, tool):
-    with NamedTemporaryFile() as f:
+    temp_name = None
+    try:
+        with NamedTemporaryFile(delete=False) as f:
+            temp_name = f.name
+            f.write(data)
+            f.flush()
+        return tool(temp_name)
+    finally:
+        if temp_name is not None:
+            os.remove(temp_name)""",
    ),
    RepairCase(
        name="tempfile_windows_mkstemp_cleanup",
        prompt="""patch only

src/importer.py
from tempfile import NamedTemporaryFile

def inspect_bytes(data, inspector):
    with NamedTemporaryFile() as tmp:
        tmp.write(data)
        tmp.flush()
        return inspector(tmp.name)

Windows test fails because inspector cannot open tmp.name while tmp is still open.
Make it cross-platform and always clean up the temp path.
""",
        answer="""--- a/src/importer.py
+++ b/src/importer.py
@@ -1,7 +1,16 @@
-from tempfile import NamedTemporaryFile
+import os
+from tempfile import mkstemp
 
 def inspect_bytes(data, inspector):
-    with NamedTemporaryFile() as tmp:
-        tmp.write(data)
-        tmp.flush()
-        return inspector(tmp.name)
+    fd, path = mkstemp()
+    try:
+        with os.fdopen(fd, "wb") as tmp:
+            tmp.write(data)
+        return inspector(path)
+    finally:
+        os.remove(path)""",
    ),
)


def row_for_case(case: RepairCase, index: int, repeat: int) -> dict:
    return {
        "instance_id": f"coder-repair-{case.name}-{repeat:03d}",
        "source": "coder_repair_synthetic",
        "messages": [
            {"role": "user", "content": case.prompt, "tool_calls": []},
            {"role": "assistant", "content": case.answer, "tool_calls": []},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/unsloth_agentic_24k_train_normalized_format_plus640")
    parser.add_argument("--out", default="data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair")
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--base-limit", type=int, default=0,
                        help="shuffle/select this many base rows before mixing; 0 keeps the full base dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--filter-base-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop base rows that do not look like edit-first traces with a verify tail.",
    )
    parser.add_argument("--max-first-edit-ratio", type=float, default=0.4)
    parser.add_argument("--max-read-streak", type=int, default=6)
    parser.add_argument(
        "--require-verify-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    base = load_from_disk(args.base)
    if args.base_limit > 0 and args.base_limit < len(base):
        base = base.shuffle(seed=args.seed).select(range(args.base_limit))
    if args.filter_base_quality:
        before = len(base)

        def keep_quality(example: dict) -> bool:
            return trace_should_keep(
                example["messages"],
                max_first_edit_ratio=args.max_first_edit_ratio,
                max_read_streak=args.max_read_streak,
                require_verify_tail=args.require_verify_tail,
            )

        base = base.filter(keep_quality, desc="filter-base-quality")
        print(f"base_quality_rows={len(base)}/{before}")
    rows = []
    for repeat in range(args.repeat):
        for index, case in enumerate(REPAIR_CASES):
            rows.append(row_for_case(case, index, repeat))

    repair = Dataset.from_list(rows, features=base.features)
    mixed = concatenate_datasets([base, repair])
    mixed.save_to_disk(args.out)
    print(f"base_rows={len(base)}")
    print(f"repair_rows={len(repair)}")
    print(f"total_rows={len(mixed)}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
