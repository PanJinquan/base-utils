#!/usr/bin/env python3
import argparse
import ast
import builtins
import io
import keyword
import random
import tokenize
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

alphabet = "abcdefghijklmnopqrstuvwxyz"

BUILTIN_NAMES = set(dir(builtins))


def remove_comments(source: str) -> str:
    tokens: List[tokenize.TokenInfo] = []
    reader = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(reader):
        if tok.type == tokenize.COMMENT:
            continue
        tokens.append(tok)
    return tokenize.untokenize(tokens)


def remove_docstring_if_present(body: List[ast.stmt]) -> List[ast.stmt]:
    if not body:
        return body
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return body[1:]
    return body


def extract_param_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> Set[str]:
    args = node.args
    names: Set[str] = set()
    for item in args.posonlyargs + args.args + args.kwonlyargs:
        names.add(item.arg)
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


class DirectLocalCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.locals: Set[str] = set()
        self.global_names: Set[str] = set()
        self.nonlocal_names: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.locals.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.locals.add(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.locals.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.locals.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.locals.add(node.name)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.locals.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.locals.add(node.name)
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.locals.add(node.rest)
        self.generic_visit(node)


class NestedNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)


class FunctionScopedRenamer(ast.NodeTransformer):
    def __init__(self, rename_map: Dict[str, str]) -> None:
        self.rename_map = rename_map

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        # `except Exception as e` stores alias name on node.name (str), not ast.Name.
        if node.name and node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        if node.type:
            node.type = self.visit(node.type)
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_MatchAs(self, node: ast.MatchAs) -> ast.AST:
        if node.name and node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        if node.pattern:
            node.pattern = self.visit(node.pattern)
        return node

    def visit_MatchStar(self, node: ast.MatchStar) -> ast.AST:
        if node.name and node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        return node

    def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.AST:
        if node.rest and node.rest in self.rename_map:
            node.rest = self.rename_map[node.rest]
        node.keys = [self.visit(key) if key is not None else None for key in node.keys]
        node.patterns = [self.visit(pattern) for pattern in node.patterns]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return node


class ObfuscatorTransformer(ast.NodeTransformer):
    def __init__(self, seed: int = 9527, junk_level: int = 2) -> None:
        self.random = random.Random(seed)
        self.junk_level = max(0, junk_level)
        self.used_symbol_names: Set[str] = set()
        self.dead_code_counter = 0

    def _next_symbol(self) -> str:
        # Generate visually noisy names to reduce readability.
        while True:
            # length = self.random.randint(2, 6)
            length = 4
            candidate = "".join(self.random.choice(alphabet) for _ in range(length))
            if candidate not in self.used_symbol_names:
                self.used_symbol_names.add(candidate)
                return candidate

    def _next_dead_name(self, blocked: Set[str]) -> str:
        while True:
            self.dead_code_counter += 1
            candidate = f"_obf_dead_{self.dead_code_counter:04d}"
            if candidate not in blocked:
                return candidate

    def _collect_nested_names(self, body: Iterable[ast.stmt]) -> Set[str]:
        nested = NestedNameCollector()
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nested.visit(stmt)
        return nested.names

    def _build_rename_map(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Dict[str, str]:
        collector = DirectLocalCollector()
        for stmt in node.body:
            collector.visit(stmt)

        params = extract_param_names(node)
        nested_names = self._collect_nested_names(node.body)
        direct_nested_def_names = {
            stmt.name
            for stmt in node.body
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

        reserved = set(keyword.kwlist) | BUILTIN_NAMES | params | collector.global_names | collector.nonlocal_names
        reserved.add("self")
        reserved.add("cls")
        reserved.update(direct_nested_def_names)

        # Avoid obfuscating symbols that may be captured in nested scopes.
        candidates = sorted(name for name in collector.locals if name not in reserved and name not in nested_names)

        rename_map: Dict[str, str] = {}
        for name in candidates:
            new_name = self._next_symbol()
            while new_name in reserved or new_name in rename_map.values():
                new_name = self._next_symbol()
            rename_map[name] = new_name
        return rename_map

    def _inject_dead_code(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.junk_level <= 0 or not node.body:
            return

        existing_names = extract_param_names(node)
        local_collector = DirectLocalCollector()
        for stmt in node.body:
            local_collector.visit(stmt)
        existing_names.update(local_collector.locals)

        dead_var = self._next_dead_name(existing_names)
        dead_msg = f"noise_{self.random.randint(1000, 9999)}"

        snippets = [
            ast.parse(f"{dead_var} = 0").body[0],
            ast.parse("if False:\n    raise RuntimeError('unreachable')").body[0],
            ast.parse(
                f"try:\n    {dead_var} += {self.random.randint(1, 5)}\nexcept Exception:\n    pass"
            ).body[0],
            ast.parse(f"if {dead_var} == -1:\n    print('{dead_msg}')").body[0],
        ]

        node.body = snippets[: min(self.junk_level, len(snippets))] + node.body

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        node.body = remove_docstring_if_present(node.body)

        rename_map = self._build_rename_map(node)
        if rename_map:
            renamer = FunctionScopedRenamer(rename_map)
            updated_body = []
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    updated_body.append(stmt)
                else:
                    updated_body.append(renamer.visit(stmt))
            node.body = updated_body

        # self._inject_dead_code(node)

        # Continue to process nested class/function bodies.
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = remove_docstring_if_present(node.body)
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = remove_docstring_if_present(node.body)
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._process_function(node)


def validate_python_source(source: str, source_label: str) -> Tuple[bool, str]:
    try:
        parsed = ast.parse(source)
        compile(parsed, source_label, "exec")
    except SyntaxError as err:
        return False, f"{source_label}: syntax error at line {err.lineno}, col {err.offset}: {err.msg}"
    except Exception as err:  # pragma: no cover
        return False, f"{source_label}: compile failed: {err}"
    return True, ""


def obfuscate_source(source: str, seed: int = 9527, junk_level: int = 2) -> str:
    no_comment_source = remove_comments(source)
    tree = ast.parse(no_comment_source)
    transformer = ObfuscatorTransformer(seed=seed, junk_level=junk_level)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    result = ast.unparse(new_tree)

    ok, msg = validate_python_source(result, "<obfuscated>")
    if not ok:
        raise SyntaxError(msg)
    return result + "\n"


def build_output_path(input_path: Path, output_root: Path, input_root: Path) -> Path:
    if input_path.is_file():
        if output_root.suffix == ".py":
            return output_root
        return output_root / f"{input_path.stem}.py"
    relative = input_path.relative_to(input_root)
    return output_root / relative


def _normalize_excludes(excludes: List[str]) -> List[str]:
    normalized: List[str] = []
    for item in excludes:
        value = item.strip().replace("\\", "/").strip("/")
        if value:
            normalized.append(value)
    return normalized


def _is_excluded(py_file: Path, root: Path, excludes: List[str]) -> bool:
    if not excludes:
        return False

    rel_path = py_file.relative_to(root).as_posix()
    rel_dir = py_file.parent.relative_to(root).as_posix()
    rel_parts = py_file.relative_to(root).parts
    dir_parts = py_file.parent.relative_to(root).parts

    for rule in excludes:
        # "a/b" means relative folder path prefix from input root.
        if "/" in rule:
            if rel_dir == rule or rel_dir.startswith(f"{rule}/"):
                return True
            continue

        # "cache" means any folder segment named "cache".
        if rule in dir_parts:
            return True

        # Allow explicit file exclusion when users pass a file name.
        if rel_path == rule or rel_path.endswith(f"/{rule}") or (rel_parts and rel_parts[-1] == rule):
            return True

    return False


def iter_python_files(path: Path, excludes: List[str] | None = None) -> Iterable[Path]:
    excludes = _normalize_excludes(excludes or [])
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return

    root = path.resolve()
    for py_file in path.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        if _is_excluded(py_file.resolve(), root, excludes):
            continue
        yield py_file


def process_path(
        src_path: Path,
        out_path: Path,
        in_place: bool,
        seed: int,
        junk_level: int,
        excludes: List[str] | None = None,
) -> Tuple[int, int]:
    src_path = src_path.resolve()
    out_path = out_path.resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"input path not found: {src_path}")

    files = list(iter_python_files(src_path, excludes=excludes))
    if not files:
        return 0, 0

    ok_count = 0
    fail_count = 0

    input_root = src_path.parent if src_path.is_file() else src_path

    for file_path in files:
        try:
            code = file_path.read_text(encoding="utf-8")
            obf_code = obfuscate_source(code, seed=seed, junk_level=junk_level)

            target = file_path if in_place else build_output_path(file_path, out_path, input_root)
            print(f"[PROCESS] {file_path}-->{target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(obf_code, encoding="utf-8")

            ok_count += 1
        except Exception as err:
            fail_count += 1
            print(f"[FAIL] {file_path}: {err}")

    return ok_count, fail_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Python code obfuscator: remove comments/docstrings, obfuscate local vars, inject dead code."
    )
    parser.add_argument("input", help="Input Python file or directory")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output file or directory. If omitted: <file>_obf.py or <dir>_obf/",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source files directly",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9527,
        help="Random seed for deterministic obfuscation",
    )
    parser.add_argument(
        "--junk-level",
        type=int,
        default=2,
        help="How many dead-code snippets to inject into each function (0-4)",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        nargs='+',
        default=[],
        help="Exclude folder/file from obfuscation. Can be repeated, e.g. -e venv tests/data",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}.py")
    return input_path.with_name(f"{input_path.name}_obf")


def main() -> int:
    args = parse_args()
    print(args)
    src_path = Path(args.input)

    if args.in_place:
        out_path = src_path
    else:
        out_path = Path(args.output) if args.output else default_output_path(src_path)

    ok_count, fail_count = process_path(
        src_path=src_path,
        out_path=out_path,
        in_place=args.in_place,
        seed=args.seed,
        junk_level=args.junk_level,
        excludes=args.exclude,
    )

    if ok_count == 0 and fail_count == 0:
        print("No Python files found.")
        return 1

    print(f"Done. success={ok_count}, failed={fail_count}")
    if not args.in_place:
        print(f"Output: {out_path.resolve()}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    """
    单文件输出到指定文件： python python_obfuscator.py input.py -o output_obf.py
    目录批量输出到新目录： python python_obfuscator.py ./src -o ./src_obf
    原地覆盖（谨慎）    ： python python_obfuscator.py ./src --in-place
    调整无效代码注入强度： python python_obfuscator.py ./src -o ./src_obf --junk-level 3 --seed 123
    """
    raise SystemExit(main())
