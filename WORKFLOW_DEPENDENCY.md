# Scientific-Workflow dependency

This application consumes a separate Scientific-Workflow fork:

- Fork: `https://github.com/wtrhacker/Scientific-Workflow`
- Fork branch: `feature/external-tools`
- Tested fork commit: `a426683b44e3106374f159ecd15d4ca666642cfd`
- Upstream baseline: `c6e788eb4a8e2e8c52cd1a4260b7949fe8a95e2d`
- Required Rust feature: `external-tools`

`native/Cargo.toml` pins this published Git revision directly:

```toml
scientific-workflow = { git = "https://github.com/wtrhacker/Scientific-Workflow", rev = "a426683b44e3106374f159ecd15d4ca666642cfd", features = ["external-tools"] }
```

The repositories have separate Git histories. Cargo fetches the pinned fork
during a source build; no sibling checkout or vendored Workflow source is used.
Update the manifest, lockfile, and this note together when changing the dependency.
Python wheels include the compiled manager and require neither checkout nor
Rust at runtime.
