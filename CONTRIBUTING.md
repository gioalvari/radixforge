# Contributing

Thanks for your interest! Issues and pull requests are welcome.

## Development setup

```bash
git clone --recursive <repo-url>
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(sysctl -n hw.ncpu)"
ctest --test-dir build --output-on-failure
```

Requires macOS on Apple Silicon, Xcode Command Line Tools and CMake.
`llama.cpp` is a pinned git submodule; bump it in a dedicated PR.

## Pull requests

- Keep each PR focused on one change and add or update tests for it.
- Make sure the build has no new warnings and `ctest` passes.
- Include benchmark numbers (hardware, model, command) for performance changes.
- Use short, imperative commit messages (e.g. `fix: handle empty prompt`).
